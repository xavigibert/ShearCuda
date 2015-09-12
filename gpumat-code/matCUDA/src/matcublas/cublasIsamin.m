% cublasIsamin - Wrapper to CUBLAS cublasIsamin function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasIsamin function. Original function dec-
% laration:
% 
% int
% cublasIsamin (int n, const float *x, int incx)
% 
% 
% EXAMPLE
% 
% N = 10;
% A = GPUsingle(rand(1,N));
% 
% Isamin = cublasIsamin(N, getPtr(A), 1);
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
% 
% [value, Isamin_mat] = min(single(A));
% compareArrays(Isamin, Isamin_mat, 1e-6);
