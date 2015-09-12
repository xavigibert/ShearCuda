% cublasSaxpy - Wrapper to CUBLAS cublasSaxpy function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasSaxpy function. Original function dec-
% laration:
% 
% void
% cublasSaxpy
% (int n, float alpha, const float *x, int incx, float *y,
%  int incy)
% 
% 
% EXAMPLE
% 
% N = 10;
% A = GPUsingle(rand(1,N));
% B = GPUsingle(rand(1,N));
% 
% alpha = 2.0;
% Saxpy_mat = alpha * single(A) + single(B);
% 
% cublasSaxpy(N, alpha, getPtr(A), 1, getPtr(B), 1);
% 
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
% 
% 
% compareArrays(Saxpy_mat, single(B), 1e-6);
