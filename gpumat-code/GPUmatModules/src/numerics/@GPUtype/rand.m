% rand - GPU pseudorandom generator
% 
% SYNTAX
% 
% rand(N,GPUsingle)
% rand(M,N,GPUsingle)
% rand([M,N],GPUsingle)
% rand(M,N,P,...?,GPUsingle)
% rand([M N P ...],GPUsingle)
% 
% rand(N,GPUdouble)
% rand(M,N,GPUdouble)
% rand([M,N],GPUdouble)
% rand(M,N,P,...?,GPUdouble)
% rand([M N P ...],GPUdouble)
% 
% 
% MODULE NAME
% RAND
% 
% DESCRIPTION
% rand(N,GPUsingle) is an N-by-N GPU matrix of values generated
% with a pseudorandom generator (uniform distribution).
% rand(M,N,GPUsingle) or rand([M,N],GPUsingle) is an M-by-N
% GPU matrix.
% rand(M,N,P,...,GPUsingle) or rand([M N P ...,GPUsingle])
% is an M-by-N-by-P-by-... GPU array of single precision values.
% rand(M,N,P,...,GPUdouble) or rand([M N P ...,GPUdouble])
% is an M-by-N-by-P-by-... GPU array of double precision values.
% Compilation supported
% 
% EXAMPLE
% 
% A   =   rand(10,GPUsingle)
% B   =   rand(10, 10,GPUsingle)
% C   =   rand([10 10],GPUsingle)
% A   =   rand(10,GPUdouble)
% B   =   rand(10, 10,GPUdouble)
% C   =   rand([10 10],GPUdouble)
