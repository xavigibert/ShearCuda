% cublasIsamax - Wrapper to CUBLAS cublasIsamax function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasIsamax function. Original function
% declaration:
% 
% int
% cublasIsamax (int n, const float *x, int incx)
% 
% Mapping:
% 
% RET = cublasIsamax(N, d_A, INCX)
% N    -> int n
% d_A -> void **devicePtr
% INCX -> int incx
% 
% RET -> cublasIsamax result
% 
% 
% EXAMPLE
% 
% N = 10;
% A = GPUsingle(rand(1,N));
% 
% Isamax = cublasIsamax(N, getPtr(A), 1);
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
% 
% [value, Isamax_mat] = max(single(A));
% compareArrays(Isamax, Isamax_mat, 1e-6);
