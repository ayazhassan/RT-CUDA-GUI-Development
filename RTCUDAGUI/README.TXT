============================
RCUDA Installation and Setup
============================

Pre-Requisites
--------------
	* Java JDK/JRE 1.7 or later, it can be downloaded from the following link:
(www.oracle.com/technetwork/java/javase/downloads/index.html)
	* NVIDIA CUDA Toolkit, it can be downloaded from the following link:
(https://developer.nvidia.com/cuda-toolkit)

Package Extraction
-----------------------
To extract RCUDA package, run the following command:

unzip RCUDACompiler.zip

===========
RCUDA Usage
===========

Input
-----
RCUDA compiler requires following files to be created in src folder:

	* kernel.c: This file contains the C function that needs to be converted as CUDA kernel to run on GPU device. The function should follow the ANSI C standard with required parameters and no return type. This function implements the C-Loop structure to be partitioned among multiple CUDA blocks and threads to compute the required results. Following is the syntax of the function definition:

void func_name(type param, ...){
	<function body>
}

	* main.c: This file contains the C main function that implements the user input, data allocation, initialization, function call (defined in kernel.c), and output of the program following the ANSI C standard. All the arrays that are going to be used by the kernel function should be allocated dynamically using C malloc() function. To run the kernel on a particular GPU device, user should use the function cudaSetDevice(GPU_ID) before calling the kernel function where GPU_ID is the id of the GPU device available in the system.

	* config.txt: This file contains the configurations for different parameters used by the compiler for optimizations and final code generation. Following are the list of parameters that need to be defined in configuration:
		- LOOP_COLLAPSING: Enabled(1)/Disabled(0) loop collapsing optimization. It is only applicable to the kernel with 2D resultant matrix and having two nested loops in the computations.
		- BLOCK_SKEW: Enabled(1)/Disabled(0) block skewing optimization. It is only applicable to the kernel with 2D resultant matrix. It increases the thread access locality by merging multiple resultant blocks horizontally to one thread block.
		- PREFETCHING: Enabled(1)/Disabled(0) prefetching optimization. It is only applicable to the kernel having 2D matrix in the computation that need to be tiled to store in shared memory.
		- PREFETCHED_ARRAYS: List of array variables that need to be tiled. This is only applied if PREFETCHING is enabled.
		- NON_PREFETCHED_ARRAYS: List of array variables that should be ignored for tiling. This is only applied if PREFTECHING is enabled.
		- DATA_TYPE: It defines the data type of the arrays in the computation and resultant that need to be stored in GPU memory.
		- KERNEL_NAMES: List of function names that need to be converted as CUDA kernels that are defined in kernel.c file.
		- 2DMATRIX: It is set to 1 for 2D resultant matrix and 0 for 1D.
		- ROW_DIM: This is the leading dimension of the matrices used in computation.
		- MAX_BLOCKSIZE: Upper bound of BLOCKSIZE to be analyzed by RCUDA Parameter Tunner. This should be less than or equal to the maxmimum possible thread block size of the underlying GPU compute capability. To check all possible block size based on the underlying GPU architecture automatically, set this value to 0.
		- MAX_MERGE_LEVEL: Upper bound of MERGE_LEVEL to be analyzed by RCUDA Parameter Tunner. This should be less than or equal to the MAX_BLOCKSIZE.
		- MAX_SKEW_LEVEL: Upper bound of SKEW_LEVEL to be analyzed by RCUDA Parameter Tunner. This should be less than or equal to the MAX_BLOCKSIZE.
		- MIN_BLOCKSIZE: Lower bound of BLOCKSIZE to be analyzed by RCUDA Parameter Tunner. This should be greater than or equal to 1 and less than or equal to the MAX_BLOCKSIZE.
		- MIN_MERGE_LEVEL: Lower bound of MERGE_LEVEL to be analyzed by RCUDA Parameter Tunner. This should be greater than or equal to 1 and less than or equal to the MAX_MERGE_LEVEL.
		- MIN_SKEW_LEVEL: Lower bound of SKEW_LEVEL to be analyzed by RCUDA Parameter Tunner. This should be greater than or equal to 1 and less than or equal to the MAX_SKEW_LEVEL.

Following is an example of a configuration file:

LOOP_COLLAPSING=1
BLOCK_SKEW=1
PREFETCHING=0
PREFETCHED_ARRAYS=A
NON_PREFETCHED_ARRAYS=B
DATA_TYPE=float
KERNEL_NAMES=matrix_scale
2DMATRIX=1
ROW_DIM=N
MAX_BLOCKSIZE=0
MAX_MERGE_LEVEL=8
MAX_SKEW_LEVEL=2
MIN_BLOCKSIZE=32
MIN_MERGE_LEVEL=1
MIN_SKEW_LEVEL=1

Execution
---------
To run the compiler from the command line, go to the dist folder and type the following:

java -jar "RCUDATranslator.jar" 

Output
------
RCUDA generates following files in the output folder:

	* kernel.cu: This file contains the converted CUDA kernels.
	
	* main.cu: This file contains the main program that calls CUDA kernels.
		
	* Header Files: The compiler generates three header files params.h, rcudacublas.h, and rcudacusparse.h that are included in the main.cu.

	* Makefile: For compilation with make program, it generates Makefile and the dependent findcudalib.mk files.
