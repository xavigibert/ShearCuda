% cublasSgemm - Wrapper to CUBLAS cublasSgemm function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasSgemm function. Original function
% declaration:
% 
% void
% cublasSgemm
% (char transa, char transb, int m, int n, int k,
%  float alpha, const float *A, int lda,
%  const float *B, int ldb, float beta,
%  float *C, int ldc)
% 
% 
% EXAMPLE
% 
% N   =   10;
% A   =   GPUsingle(rand(N,N));
% B   =   GPUsingle(rand(N,N));
% C   =   zeros(N,N,GPUsingle);
% 
% alpha = 2.0;
% beta = 0.0;
% 
% opA = 'n';
% opB = 'n';
% 
% cublasSgemm(opA, opB, N, N, N, ...
%   alpha, getPtr(A), N, getPtr(B), ...
%   N, beta, getPtr(C), N);
% 
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
