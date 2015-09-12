% cublasAlloc - Wrapper to CUBLAS cublasAlloc function
% 
% SYNTAX
% 
% [status d_A] = cublasAlloc(N,SIZE,d_A);
% N - number of elements to allocate
% SIZE - size of the elements to allocate
% d_A - pointer to GPU memory
% status - CUBLAS status
% d_A - pointer to GPU memory
% 
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasAlloc function.
% Original function declaration:
% 
% cublasStatus
% cublasAlloc (int n, int elemSize, void **devicePtr)
% 
% Mapping:
% 
% [status    d_A] = cublasAlloc(N, SIZE, d_A)
% N    ->    int n
% SIZE ->    int elemSize
% d_A ->     void **devicePtr
% 
% status -> cublasStatus
% 
% 
% EXAMPLE
% 
% N = 10;
% SIZEOF_FLOAT = sizeoffloat();
% % GPU variable d_A
% d_A = 0;
% [status d_A] = cublasAlloc(N,SIZEOF_FLOAT,d_A);
% ret = cublasCheckStatus( status, ...
%   '!!!! device memory allocation error (d_A)');
