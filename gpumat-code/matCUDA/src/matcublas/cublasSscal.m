% cublasSscal - Wrapper to CUBLAS cublasSscal function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasSscal function.
% Original function declaration:
% 
% void
% sscal (int n, float alpha, float *x, int incx)
% 
% 
% EXAMPLE
% 
% N = 10;
% A = GPUsingle(rand(1,N));
% 
% alpha = 1/10.0;
% A_mat = single(A)*alpha;
% cublasSscal(N, alpha, getPtr(A), 1);
% 
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
