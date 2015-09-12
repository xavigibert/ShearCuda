% cublasScopy - Wrapper to CUBLAS cublasScopy function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasScopy function. Original function dec-
% laration:
% 
% void
% cublasScopy
% (int n, const float *x, int incx, float *y, int incy)
% 
% 
% EXAMPLE
% 
% N = 10;
% A = GPUsingle(rand(1,N));
% B = GPUsingle(rand(1,N));
% 
% cublasScopy(N, getPtr(A), 1, getPtr(B), 1);
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
% 
% compareArrays(single(A), single(B), 1e-6);
