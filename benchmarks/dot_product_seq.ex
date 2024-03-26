[arg] = System.argv()
n = String.to_integer(arg)

a = Matrex.new([Enum.to_list(0..(n - 1))])
b = Matrex.new([Enum.to_list(0..(n - 1))])

prev = System.monotonic_time()
c = Matrex.dot_nt(a, b)
next = System.monotonic_time()

IO.puts "Elixir\t#{n}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

IO.puts "FINAL RESULTADO: #{Enum.at(Matrex.row_to_list(c, 1), 0)}"
