defmodule GPotion do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/cpu_nifs', 0)
  #    #IO.puts("ok")
  end
  defp gen_para(p,:matrex) do
    "float *#{p}"
  end
  defp gen_para(p,:float) do
    "float #{p}"
  end
  defp gen_para(p,:int) do
    "int #{p}"
  end

  def format_c_code(file_path) do
    command = "clang-format"
    args = ["-i", file_path] # `-i` edits files in place

    case System.cmd(command, args) do
      {_, 0} ->
        {:ok, "File formatted successfully."}
      {error_message, _} ->
        {:error, error_message}
    end
  end

  defmacro gpotion(header, do: body) do
   {fname, comp_info, para} = header

    caller_st = __CALLER__
    module_name = to_string caller_st.module
    #IO.puts module_name

   {param_list,types_para,is_typed,inf_types} = if is_list(List.last(para)) do
      types_para = List.last(para)
      param_list = para
        |> List.delete_at(length(para)-1)
        |> Enum.map(fn({p, _, _}) -> p end)
        |> Enum.zip(types_para)
        |> Enum.map(fn({p,t}) -> gen_para(p,t) end)
        |> Enum.join(", ")
      {param_list,types_para,true,%{}}
    else
      types = para
      |> Enum.map(fn({p, _, _}) -> p end)
      |> Map.new(fn x -> {x,:none} end)
      |> GPotion.TypeInference.infer_types(body)

      param_list = para
      |> Enum.map(fn {p, _, _}-> gen_para(p,Map.get(types,p)) end)
      |> Enum.join(", ")

      types_para = para
      |>  Enum.map(fn {p, _, _}-> Map.get(types,p) end)
     {param_list,types_para,false,types}
   end

   #IO.puts("C Backend:\n")
   k = GPotion.CBackend.c_code_generation(body,fname, param_list,inf_types,is_typed)
   access_func = GPotion.CBackend.generate_access_function(fname, length(types_para), Enum.reverse(types_para))
   #IO.inspect(access_func)


   #accessfunc = GPotion.CudaBackend.gen_kernel_call(fname,length(types_para),Enum.reverse(types_para))
   file = File.open!("c_src/#{module_name}.c", [:write])
   IO.write(file, "#include \"erl_nif.h\"\n#include <pthread.h>\n\n" <> GPotion.CBackend.generate_dim3_structure() <> k <> "\n\n" <> access_func)
   File.close(file)

   format_c_code("c_src/#{module_name}.c")
   #IO.puts k
   #IO.puts accessfunc
   #para = if is_list(List.last(para)) do List.delete_at(para,length(para)-1) else para end
   #para = para
   # |> Enum.map(fn {p, b, c}-> {String.to_atom("_" <> to_string(p)),b,c} end)

  {_result, _errcode} = System.cmd("gcc",
  [
  "-lpthread",
  "-o",
  "priv/#{module_name}.so",
  "-fpic",
  "-shared",
  "c_src/#{module_name}.c"
  ], stderr_to_stdout: true)
  File.rename("c_src/#{module_name}.c","c_src/#{module_name}_gp.c")

  quote do
  end
end

  def create_ref_nif(_matrex) do
    raise "NIF create_ref_nif/1 not implemented"
  end
  def new_pinned_nif(_list,_length) do
    raise "NIF new_pinned_nif/1 not implemented"
  end
  def new_gmatrex_pinned_nif(_array) do
    raise "NIF new_gmatrex_pinned_nif/1 not implemented"
  end
  def new_pinned(list) do
    size = length(list)
    {new_pinned_nif(list,size), {1,size}}
  end
  def new_gmatrex(%Matrex{data: matrix} = a) do
    ref=create_ref_nif(matrix)
    {ref, Matrex.size(a)}
  end
  def new_gmatrex({array,{l,c}}) do
    ref=new_gmatrex_pinned_nif(array)
    {ref, {l,c}}
  end

  def new_gmatrex(r,c) do
    ref=new_ref_nif(c)
    {ref, {r,c}}
    end

  def new_ref_nif(_matrex) do
    raise "NIF new_ref_nif/1 not implemented"
  end
  def synchronize_nif() do
    raise "NIF new_ref_nif/1 not implemented"
  end
  def synchronize() do
    synchronize_nif()
  end
  def new_ref(size) do
  ref=new_ref_nif(size)
  {ref, {1,size}}
  end
  def get_matrex_nif(_ref,_rows,_cols) do
  raise "NIF get_matrex_nif/1 not implemented"
  end
  def get_gmatrex({ref,{rows,cols}}) do
  %Matrex{data: get_matrex_nif(ref,rows,cols)}
  end

  def load_kernel_nif(_module,_fun) do
    raise "NIF new_ref_nif/2 not implemented"
  end
  def load(kernel) do
    case Macro.escape(kernel) do
      {:&, [],[{:/, [], [{{:., [], [module, kernelname]}, [no_parens: true], []}, _nargs]}]} ->


        #IO.puts module
        #IO.puts kernelname
        #raise "hell"

        GPotion.load_kernel_nif(to_charlist(module),to_charlist(kernelname))

      _ -> raise "GPotion.build: invalid kernel"
    end
  end
  def spawn_nif(_k,_t,_b,_l) do
    raise "NIF spawn_nif/1 not implemented"
  end
  def spawn(k,t,b,l) when is_function(k) do
    load(k)
    spawn_nif(k,t,b,Enum.map(l,&get_ref/1))
  end
  def spawn(k,t,b,l) do
    spawn_nif(k,t,b,Enum.map(l,&get_ref/1))
  end
  def get_ref({ref,{_rows,_cols}}) do
    ref
  end
  def get_ref(e) do
    e
  end
end
