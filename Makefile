all: priv/gpu_nifs.so 

priv/gpu_nifs.so: c_src/gpu_nifs.cu
	gcc --shared -g --compiler-options '-fPIC' -o priv/gpu_nifs.so c_src/nifs.cu

bmp: c_src/bmp_nifs.cu 
	gcc --shared -g --compiler-options '-fPIC' -o priv/bmp_nifs.so c_src/bmp_nifs.c

clean:
	rm priv/gpu_nifs.so
