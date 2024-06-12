defmodule GPotion.CBackend do
  @moduledoc """
  Este módulo fornece funcionalidades para gerar código backend C para simular a execução de kernels CUDA em CPU.
  """

  defp get_shared_variables(ast) do
    get_shared_variables(ast, [])
  end

  defp get_shared_variables({:__block__, _, children}, acc) do
    Enum.reduce(children, acc, &get_shared_variables(&1, &2))
  end

  defp get_shared_variables({:__shared__,_ , [{{:., _, [Access, :get]}, _, [arg1,arg2]}]}, acc) do
    ["float #{generate_expression arg1}[#{generate_expression arg2}];" | acc]
  end

  defp get_shared_variables({_op, _meta, args}, acc) when is_list(args) do
    Enum.reduce(args, acc, &get_shared_variables(&1, &2))
  end

  defp get_shared_variables(_, acc), do: acc

  defp generate_function_call(name, para, body) do
    "void #{name} (#{para}, Dim3 gridDim, Dim3 blockDim) { \n #{body} \n}"
  end

  defp generate_thread_function(para, body) do
    "void* thread_kernel (void *arg) {
      ThreadData *data = (ThreadData *)arg;\n
      #{
        Regex.replace(~r/(\w+)\s+(\w+)\[(\d+)\]/, para, "\\1 * \\2")
        |> String.split(",")
        |> Enum.map(fn p -> "#{p} = data->#{
          String.split(p, ~r/[* ]/, trim: true)
          |> Enum.at(1)
        };\n" end)}
      Dim3 threadIdx = data->threadIdx;
      Dim3 blockIdx = data->blockIdx;
      Dim3 blockDim = data->blockDim;
      Dim3 gridDim = data->gridDim;
      \n#{body}\n
      return NULL;
    }
    \n"
  end

  defp get_dimensions(ast) do
    {grid_dims, block_dims} = get_dimentions_(ast, {MapSet.new(), MapSet.new()})

    {MapSet.size(grid_dims), MapSet.size(block_dims)}
  end

  defp get_dimentions_({:__block__, _, ast_list}, acc) do
    Enum.reduce(ast_list, acc, &get_dimentions_(&1, &2))
  end
  defp get_dimentions_({op, _, args} = ast, {g, b}) when is_list(args) and op not in [:__block__, :__syncthreads] do
    case ast do
      {{:., _, [{:blockIdx, _, _}, dim]}, _, _} -> {MapSet.put(g, dim), b}
      {{:., _, [{:threadIdx, _, _}, dim]}, _, _} -> {g, MapSet.put(b, dim)}
      _ -> Enum.reduce(args, {g, b}, &get_dimentions_(&1, &2))
    end
  end
  defp get_dimentions_([head | tail], acc) do
    get_dimentions_(tail, get_dimentions_(head, acc))
  end
  defp get_dimentions_([], acc), do: acc
  defp get_dimentions_(_, acc), do: acc

  # se usar __syncthreads, tratar de forma diferente
  defp uses_thread_sync?(ast), do: check_for_thread_sync?(ast, false)

  defp check_for_thread_sync?({:__syncthreads, _, _}, _), do: true
  defp check_for_thread_sync?({_operation, _meta, args}, _) when is_list(args) do
    Enum.any?(args, &check_for_thread_sync?(&1, false))
  end
  defp check_for_thread_sync?([head | tail], acc) when is_list(head) or is_tuple(head) do
    check_for_thread_sync?(head, acc) || Enum.any?(tail, &check_for_thread_sync?(&1, false))
  end
  defp check_for_thread_sync?([], acc), do: acc
  defp check_for_thread_sync?(_, acc), do: acc

  # se usar __shared__, tratar de forma diferente
  defp uses_shared_variables?(ast), do: check_for_shared_variables?(ast, false)

  defp check_for_shared_variables?({:__syncthreads, _, _}, _), do: true
  defp check_for_shared_variables?({_operation, _meta, args}, _) when is_list(args) do
    Enum.any?(args, &check_for_shared_variables?(&1, false))
  end
  defp check_for_shared_variables?([head | tail], acc) when is_list(head) or is_tuple(head) do
    check_for_shared_variables?(head, acc) || Enum.any?(tail, &check_for_shared_variables?(&1, false))
  end
  defp check_for_shared_variables?([], acc), do: acc
  defp check_for_shared_variables?(_, acc), do: acc

  defp generate_shared_vars_alloc(shared_vars, size) do
    Enum.reverse(shared_vars)
    |> Enum.map(fn v ->
    Regex.replace(~r/(\w+)\s+(\w+)\[(\d+)\]/, v, "\\1 **\\2")
    |> String.replace(";", "")
    |> Kernel.<>(" = (float **)malloc(#{size} * sizeof(float *));")
    end)
  end

  defp generate_function_variables(b_dim, g_dim, shared, shared_vars, sync) do
    nt = case b_dim do
      0 -> "int numThreads = 1;"
      1 -> "int numThreads = blockDim.x;"
      2 -> "int numThreads = blockDim.x * blockDim.y;"
      3 -> "int numThreads = blockDim.x * blockDim.y * blockDim.z;"
      _ -> raise "Invalid dimension!"
    end

    s_var = case g_dim do
      0 -> ""
      1 -> "gridDim.x"
      2 -> "gridDim.x * gridDim.y"
      3 -> "gridDim.x * gridDim.y * gridDim.z"
      _ -> raise "Invalid dimension!"
    end

    """
    #{nt}

    #{if (sync), do: generate_init_barrier_variable()}

    int totalThreads = #{s_var} * numThreads;

    pthread_t *threads = (pthread_t *)malloc(totalThreads * sizeof(pthread_t));
    #{generate_init_thread_variable()}

    ThreadData *threadData = (ThreadData *)malloc(totalThreads * sizeof(ThreadData));
    #{generate_init_thread_data_variable()}


    #{generate_shared_vars_alloc(shared_vars, s_var)}

    Dim3 blockIdx;
    Dim3 threadIdx;

    #{if shared, do: "int bid = 0;"}
    int tid = 0;
    \n
    """
  end

  defp generate_simulation_loop(body, dims, index_variable, dimension_variable) do
    case dims do
      0 -> body
      1 -> simulate_1d_loop_structure(index_variable, dimension_variable, body)
      2 -> simulate_2d_loop_structure(index_variable, dimension_variable, body)
      3 -> simulate_3d_loop_structure(index_variable, dimension_variable, body)
      _ -> raise "Invalid dimension!"
    end
  end

  defp simulate_1d_loop_structure(index_variable, dimension_variable, body) do
    """
    for (#{index_variable}.x = 0; #{index_variable}.x < #{dimension_variable}.x; ++#{index_variable}.x) {\n
      #{body}\n
    }\n
    """
  end

  defp simulate_2d_loop_structure(index_variable, dimension_variable, body) do
    """
    for (#{index_variable}.y = 0; #{index_variable}.y < #{dimension_variable}.y; ++#{index_variable}.y) {
      #{simulate_1d_loop_structure(index_variable, dimension_variable, body)}
    }\n
    """
  end

  defp simulate_3d_loop_structure(index_variable, dimension_variable, body) do
    """
    for (#{index_variable}.z = 0; #{index_variable}.z < #{dimension_variable}.z; ++#{index_variable}.z) {
      #{simulate_2d_loop_structure(index_variable, dimension_variable, body)}
    }\n
    """
  end

  defp generate_kernel_simulation_loop(body, grid_dimension, block_dimension, shared, s_vars) do
    thread_loop = body
    |> generate_simulation_loop(block_dimension, "threadIdx", "blockDim")
    |> Kernel.<>(generate_pthread_join())

    s_vars = Enum.reverse(s_vars)
    |> Enum.map(fn v ->
    var = Regex.replace(~r/(\w+)\s+(\w+)\[(\d+)\]/, v, "\\1 \\2")
    |> String.replace(";", "")
    |> String.split(" ")
    |> Enum.at(1)

    """
    #{var}[bid] = (float *)malloc(numThreads * sizeof(float));
    if (#{var}[bid] == NULL) {
      fprintf(stderr, "Error allocating memory for #{var}[%d]\\n", bid);
      exit(1);
    }
    """
    end)
    |> Enum.join("\n")
    |> Kernel.<>("\n")


    if shared do
      s_mem = s_vars <> "\n" <> thread_loop <> "bid++;"

      s_mem
      |> generate_simulation_loop(grid_dimension, "blockIdx", "gridDim")
    else
      s_mem = s_vars <> "\n" <> thread_loop

      s_mem
      |> generate_simulation_loop(grid_dimension, "blockIdx", "gridDim")
    end
  end

  def c_code_generation(body, fname, param_list, types, is_typed) do
    pid = spawn_link(fn -> types_server([], types, is_typed) end)
    Process.register(pid, :types_server)

    {grid_dimension, block_dimension} = get_dimensions(body)

    code = generate_body(body)
    send(pid, {:kill})

    if uses_thread_sync?(body) do
      s_vars = get_shared_variables(body)

      param_s_list = s_vars
      |> Enum.reduce(param_list, fn v, acc -> String.replace(v, ";", "") <> ", " <> acc end)

      p_call = generate_pthread_call(s_vars, param_list, uses_shared_variables?(body))

      c_body = generate_function_variables(block_dimension, grid_dimension, true, s_vars, true)
      |>  Kernel.<>(generate_kernel_simulation_loop(p_call, grid_dimension, block_dimension, true, s_vars))
      |> Kernel.<>(generate_free_shared_vars(s_vars, grid_dimension))
      |> Kernel.<>(generate_pthread_free())
      |> Kernel.<>(generate_destroy_barrier_variable())

      generate_dim3_structure()
      <> generate_thread_structure(param_s_list)
      <> generate_pthread_barrier()
      <> generate_thread_function(param_s_list, code)
      <> generate_function_call(fname, param_list, c_body)
    else
      p_call = generate_pthread_call([], param_list, uses_shared_variables?(body))

      c_body = generate_function_variables(block_dimension, grid_dimension, false, [], false)
      |> Kernel.<>(generate_kernel_simulation_loop(p_call, grid_dimension, block_dimension, false, []))
      |> Kernel.<>(generate_pthread_free())

      generate_dim3_structure()
      <> generate_thread_structure(param_list)
      <> generate_thread_function(param_list, code)
      <> generate_function_call(fname, param_list, c_body)
    end
  end

  defp generate_pthread_barrier() do
    """
    pthread_barrier_t barrier;\n
    """
  end

  defp generate_init_barrier_variable() do
    """
    if (pthread_barrier_init(&barrier, NULL, numThreads) != 0) {
      fprintf(stderr, "Error initializing barrier\\n");
      exit(1);
    }\n
    """
  end

  defp generate_init_thread_variable() do
    """
    if (threads == NULL) {
      fprintf(stderr, "Error allocating memory for threads\\n");
      exit(1);
    }\n
    """
  end

  defp generate_init_thread_data_variable() do
    """
    if (threadData == NULL) {
      fprintf(stderr, "Error allocating memory for thread data\\n");
      exit(1);
    }
    """
  end

  defp generate_destroy_barrier_variable() do
    """
    \npthread_barrier_destroy(&barrier);\n
    """
  end

  defp generate_pthread_free() do
    """
    \nfree(threads);
    free(threadData);
    """
  end

  defp generate_free_shared_vars(s_vars, g_dim) do
    size = case g_dim do
      0 -> ""
      1 -> "gridDim.x"
      2 -> "gridDim.x * gridDim.y"
      3 -> "gridDim.x * gridDim.y * gridDim.z"
      _ -> raise "Invalid dimension!"
    end

    """
    for (int i = 0; i < #{size}; ++i) {
      #{
        Enum.reverse(s_vars)
        |> Enum.map(fn v ->
        a = Regex.replace(~r/(\w+)\s+(\w+)\[(\d+)\]/, v, "\\1 \\2")
        |> String.replace(";", "")
        |> String.split(" ")
        |> Enum.at(1)

        "free(#{a}[i]);"
        end)
        |> Enum.join("\n")
      }
    }
    """


  end

  defp generate_pthread_call(s_vars, para, shared) do
    """
    threadData[tid] = (ThreadData){
      #{if shared do
        Enum.reverse(s_vars)
        |> Enum.map( fn v ->
          Regex.replace( ~r/(\w+)\[(\d+)\]/, v, "\\1")
          |> String.replace(";", "")
          |> String.split(" ")
          |> Enum.at(1)
          |> Kernel.<>("[bid]")
        end)
        |> Enum.join(", ")
        |> Kernel.<>(",")
      end
      }
      #{p = String.split(para, ",")
      |> Enum.map(fn p ->
        String.split(p, ~r/[* ]/, trim: true)
        |> Enum.at(1)
      end)
      |> Enum.join(",")
      Regex.replace( ~r/(\w+)\[(\d+)\]/, p, "\\1")}, threadIdx, blockIdx, blockDim, gridDim};\n
    if (pthread_create(&threads[tid], NULL, thread_kernel, &threadData[tid]) != 0) {
      fprintf(stderr, "Error creating thread %d\\n", tid);
      exit(1);
    }

    tid++;
    """
  end

  defp generate_pthread_join() do
    """
    for (int i = tid - numThreads; i < tid; ++i) {
      pthread_join(threads[i], NULL);
    }

    """
  end

  defp generate_body({:__block__, pos, code}), do: generate_block({:__block__, pos, code})
  defp generate_body({:do, {:__block__, pos, code}}), do: generate_block({:__block__, pos, code})
  defp generate_body({:do, exp}), do: generate_command(exp)
  defp generate_body(body), do: generate_command(body)

  defp generate_block({:__block__, _, code}) do
    code
    |> Enum.map(&generate_command/1)
    |> Enum.join("\n")
  end

  defp generate_header_for({:in, _, [{var, _, nil}, {:range, _, [n]}]}) do
    "for(int #{var} = 0; #{var}<#{generate_expression n}; #{var}++)"
  end
  defp generate_header_for({:in, _,[{var, _, nil}, {:range, _, [argr1, argr2]}]}) do
    "for(int #{var} = #{generate_expression argr1}; #{var}<#{generate_expression argr2}; #{var}++)"
  end
  defp generate_header_for({:in, _,[{var, _, nil}, {:range, _, [argr1, argr2, step]}]}) do
    "for(int #{var} = #{generate_expression argr1}; #{var}<#{generate_expression argr2}; #{var}+=#{generate_expression step})"
  end

  defp generate_command({:for,_, [param, [body]]}) do
    generate_header_for(param) <> "{\n" <> generate_body(body) <> "\n}\n"
  end
  defp generate_command({:=, _, [arg, exp]}) do
    a = generate_expression arg
    e = generate_expression exp

    case arg do
      {{:., _, [Access, :get]}, _, [_,_]} -> "\t#{a} = #{e}\;"
      _ ->
        send(:types_server, {:check_var, a, self()})
        receive do
          {:is_typed} -> "\t#{a} = #{e}\;"
          {:type, type}-> "\t#{type} #{a} = #{e}\;"
          {:alredy_declared} -> "\t#{a} = #{e}\;"
        end
    end
  end
  defp generate_command({:if, _, if_com}), do: generate_if(if_com)
  defp generate_command({:do_while, _, [[doblock]]}), do: "do{\n" <> generate_body(doblock)
  defp generate_command({:do_while_test, _, [exp]}), do: "\nwhile("<> (generate_expression exp) <>  ");"
  defp generate_command({:while, _, [bexp, [body]]}), do: "while(" <> (generate_expression bexp) <> "){\n" <> (generate_body body) <> "\n}"
  # CRIAÇÃO DE NOVOS VETORES
  defp generate_command({{:., _, [Access, :get]}, _, [arg1, arg2]}), do: "float #{generate_expression arg1}[#{generate_expression arg2}];"
  defp generate_command({:__shared__,_ , [{{:., _, [Access, :get]}, _, [_,_]}]}), do: ""
  defp generate_command({:__syncthreads, _, _}), do: "\npthread_barrier_wait(&barrier);\n"
  defp generate_command({:return, _, _}), do: "return;"
  defp generate_command({:var, _, [{var, _, [{:=, _, [{type, _, nil}, exp]}]}]}), do: "#{to_string type} #{to_string var} = #{generate_expression exp};"
  defp generate_command({:var, _, [{var, _, [{:=, _, [type, exp]}]}]}), do: "#{to_string type} #{to_string var} = #{generate_expression exp};"
  defp generate_command({:var, _, [{var, _, [{type, _, _}]}]}), do: "#{to_string type} #{to_string var};"
  defp generate_command({:var, _, [{var, _, [type]}]}), do: "#{to_string type} #{to_string var};"
  defp generate_command({fun, _, args}) when is_list(args) do
    nargs = args
    |> Enum.map(&generate_expression/1)
    |> Enum.join(", ")

    "#{fun}(#{nargs})\;"
  end
  defp generate_command({str,_ ,_ }), do: "#{to_string str};"
  defp generate_command(number) when is_integer(number) or is_float(number), do: to_string(number)

  defp generate_expression({{:., _, [Access, :get]}, _, [arg1, arg2]}), do: "#{generate_expression arg1}[#{generate_expression arg2}]"
  defp generate_expression({{:., _, [{struct, _, nil}, field]}, _, []}), do: "#{to_string struct}.#{to_string(field)}"
  defp generate_expression({{:., _, [{:__aliases__, _, [struct]}, field]}, _, []}), do: "#{to_string struct}.#{to_string(field)}"
  defp generate_expression({op, _, args}) when op in [:+, :-, :/, :*, :<=, :<, :>, :>=, :&&, :||, :!,:!=,:==] do
    case args do
      [a1] -> "(#{to_string(op)} #{generate_expression a1})"
      [a1, a2] -> "(#{generate_expression a1} #{to_string(op)} #{generate_expression a2})"
    end
  end
  defp generate_expression({var, _, nil}) when is_atom(var), do: to_string(var)
  defp generate_expression({fun, _, args}) do
    nargs = args
    |> Enum.map(&generate_expression/1)
    |> Enum.join(", ")

    "#{fun}(#{nargs})"
  end
  defp generate_expression(number) when is_integer(number) or is_float(number), do: to_string(number)
  defp generate_expression(string) when is_binary(string), do: "\"#{string}\""

  defp generate_if([bexp, [do: then]]), do: generate_then([bexp, [do: then]])
  defp generate_if([bexp, [do: thenbranch, else: elsebranch]]) do
    generate_then([bexp, [do: thenbranch]]) <> "else{\n" <> (generate_body elsebranch) <> "\n}\n"
  end

  defp generate_then([bexp, [do: then]]), do: "if(#{generate_expression bexp})\n" <> "{\n" <> (generate_body then) <> "\n}\n"

  defp generate_dim3_structure() do
    """
    typedef struct dim3 {
      int x;
      int y;
      int z;
    } Dim3;\n
    """
  end

  defp generate_thread_structure(param_list) do
    """
    typedef struct {
      #{p = String.replace(param_list, ",", ";\n")
        Regex.replace(~r/(\w+)\s+(\w+)\[(\d+)\]/, p, "\\1 *\\2")};
      Dim3 threadIdx;
      Dim3 blockIdx;
      Dim3 blockDim;
      Dim3 gridDim;
    } ThreadData;\n
    """
  end

  def generate_access_function(kname, nargs, types) do
    generate_header(kname) <> generate_arguments(nargs, types) <> generate_call(kname, nargs)
  end

  defp generate_header(fname) do
    "void #{fname}_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
    {

      ERL_NIF_TERM list;
      ERL_NIF_TERM head;
      ERL_NIF_TERM tail;
      float **array_res;

      const ERL_NIF_TERM *tuple_blocks;
      const ERL_NIF_TERM *tuple_threads;
      int arity;

      if (!enif_get_tuple(env, argv[1], &arity, &tuple_blocks)) {
        printf (\"spawn: blocks argument is not a tuple\");
      }

      if (!enif_get_tuple(env, argv[2], &arity, &tuple_threads)) {
        printf (\"spawn:threads argument is not a tuple\");
      }
      int b1,b2,b3,t1,t2,t3;

      enif_get_int(env,tuple_blocks[0],&b1);
      enif_get_int(env,tuple_blocks[1],&b2);
      enif_get_int(env,tuple_blocks[2],&b3);
      enif_get_int(env,tuple_threads[0],&t1);
      enif_get_int(env,tuple_threads[1],&t2);
      enif_get_int(env,tuple_threads[2],&t3);

      list= argv[3];

      Dim3 gridDim;
      Dim3 blockDim;

      gridDim.x = b1;
      gridDim.y = b2;
      gridDim.z = b3;
      blockDim.x = t1;
      blockDim.y = t2;
      blockDim.z = t3;\n
    "
  end

  defp generate_call(kernelname,nargs) do
    "   #{kernelname}" <> generate_call_arguments(nargs) <> ";
    }
    "
  end

  defp generate_call_arguments(nargs), do: "(" <> generate_call_arguments_(nargs - 1) <>"arg#{nargs}, gridDim, blockDim)"
  defp generate_call_arguments_(0), do: ""
  defp generate_call_arguments_(n), do: generate_call_arguments_(n - 1) <> "arg#{n},"

  defp generate_arguments(0, _l), do: ""
  defp generate_arguments(n, []), do: generate_arguments(n - 1, []) <> generate_matrix_argument(n)
  defp generate_arguments(n, [:matrex | t]), do: generate_arguments(n - 1, t) <> generate_matrix_argument(n)
  defp generate_arguments(n, [:int | t]), do: generate_arguments(n - 1, t) <> generate_integer_argument(n)
  defp generate_arguments(n, [:float | t]), do: generate_arguments(n - 1, t) <> generate_float_argument(n)
  defp generate_arguments(n, [:double | t]), do: generate_arguments(n - 1, t) <> generate_double_argument(n)

  defp generate_matrix_argument(narg) do
    "  enif_get_list_cell(env,list,&head,&tail);
      enif_get_resource(env, head, type, (void **) &array_res);
      float *arg#{narg} = *array_res;
      list = tail;\n
    "
  end

  defp generate_integer_argument(narg) do
    "  enif_get_list_cell(env,list,&head,&tail);
      int arg#{narg};
      enif_get_int(env, head, &arg#{narg});
      list = tail;\n
    "
  end

  defp generate_float_argument(narg) do
    "  enif_get_list_cell(env,list,&head,&tail);
      double darg#{narg};
      float arg#{narg};
      enif_get_double(env, head, &darg#{narg});
      arg#{narg} = (float) darg#{narg};
      list = tail;\n
    "
  end

  defp generate_double_argument(narg) do
    "  enif_get_list_cell(env,list,&head,&tail);
      double arg#{narg};
      enif_get_double(env, head, &darg#{narg});
      list = tail;\n
    "
  end

  defp types_server(used, types, is_typed) do
    if (is_typed) do
      receive do
        {:check_var, _var, pid} ->
            send(pid, {:is_typed})
            types_server(used, types, is_typed)
        {:kill} -> :ok
      end
    else
      receive do
        {:check_var, var, pid} ->
          if (!Enum.member?(used, var)) do
            type = Map.get(types, String.to_atom(var))
            if(type == nil) do
              IO.inspect var
              IO.inspect types
              raise "Could not find type for variable #{var}. Please declare it using \"var #{var} type\""
            end
            send(pid, {:type, type})
            types_server([var | used], types, is_typed)
          else
            send(pid, {:alredy_declared})
            types_server(used, types, is_typed)
          end
        {:kill} -> :ok
      end
    end
  end
end
