run gemm_kernel.cu
Compile the code using nvcc:
nvcc -o gemm_kernel gemm_kernel.cu
Execute the compiled program:
./gemm_kernel


run Llama2.java
Download the model checkpoint file:
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
Compile the Java code:
javac Llama2.java
Execute the compiled class:
java Llama2