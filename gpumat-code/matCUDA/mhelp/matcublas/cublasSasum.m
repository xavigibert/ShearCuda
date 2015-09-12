% cublasSasum - Wrapper to CUBLAS cublasSasum function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasSasum function.
% Original function declaration:
% 
% float
% cublasSasum (int n, const float *x, int incx)
% 
% 
% EXAMPLE
% 
% N = 10;
% A = GPUsingle(rand(1,N));
% 
% Sasum = cublasSasum( N, getPtr(A), 1);
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
% 
% Sasum_mat = sum(abs(single(A)));
% compareArrays(Sasum, Sasum_mat, 1e-6);
