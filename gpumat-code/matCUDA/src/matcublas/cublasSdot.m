% cublasSdot - Wrapper to CUBLAS cublasSdot function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasSdot function. Original function dec-
% laration:
% 
% float
% cublasSdot
% (int n, const float *x, int incx, const float *y, int incy)
% 
% 
% EXAMPLE
% 
% N = 10;
% A = GPUsingle(rand(1,N));
% B = GPUsingle(rand(1,N));
% 
% Sdot_mat = sum(single(A).*single(B));
% Sdot = cublasSdot(N, getPtr(A), 1, getPtr(B), 1);
% 
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
% 
% compareArrays(Sdot_mat, Sdot, 1e-6);
