% cublasSnrm2 - Wrapper to CUBLAS cublasSnrm2 function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasSnrm2 function. Original function dec-
% laration:
% 
% float
% cublasSnrm2 (int n, const float *x, int incx)
% 
% 
% EXAMPLE
% 
% N = 10;
% A = GPUsingle(rand(1,N));
% 
% Snrm2_mat = sqrt(sum(single(A).*single(A)));
% Snrm2 = cublasSnrm2(N, getPtr(A),1);
% 
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
