% cublasCgemm - Wrapper to CUBLAS cublasCgemm function
% 
% DESCRIPTION
% Wrapper to CUBLAS cublasCgemm function. Original function
% declaration:
% 
% void cublasCgemm
% (char transa, char transb, int m, int n, int k,
%  cuComplex alpha, const cuComplex *A, int lda,
%  const cuComplex *B, int ldb, cuComplex beta,
%  cuComplex *C, int ldc)
% 
% 
% EXAMPLE
% 
% I   =   sqrt(-1);
% N   =   10;
% A   =   GPUsingle(rand(N,N) + I*rand(N,N));
% B   =   GPUsingle(rand(N,N) + I*rand(N,N));
% %   C   needs to be complex as well
% C   =   zeros(N,N,GPUsingle)*I;
% 
% % alpha is complex
% alpha = 2.0+I*3.0;
% beta = 0.0;
% 
% opA = 'n';
% opB = 'n';
% 
% cublasCgemm(opA, opB, N, N, N, ...
%   alpha, getPtr(A), N, getPtr(B), ...
%   N, beta, getPtr(C), N);
% 
% status = cublasGetError();
% ret = cublasCheckStatus( status, ...
%   '!!!! kernel execution error.');
