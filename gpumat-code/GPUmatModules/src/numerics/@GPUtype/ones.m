% ones - GPU ones array
% 
% SYNTAX
% 
% ones(N,GPUsingle)
% ones(M,N,GPUsingle)
% ones([M,N],GPUsingle)
% ones(M,N,P,...?,GPUsingle)
% ones([M N P ...],GPUsingle)
% 
% ones(N,GPUdouble)
% ones(M,N,GPUdouble)
% ones([M,N],GPUdouble)
% ones(M,N,P, ...,GPUdouble)
% ones([M N P ...],GPUdouble)
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ones(N,GPUsingle) is an N-by-N GPU matrix of ones.
% ones(M,N,GPUsingle) or ones([M,N],GPUsingle) is an M-by-N
% GPU matrix of ones.
% ones(M,N,P,...,GPUsingle) or ones([M N P ...,GPUsingle])
% is an M-by-N-by-P-by-... GPU array of ones.
% ones(M,N,P,...,GPUdouble) or ones([M N P ...,GPUdouble])
% is an M-by-N-by-P-by-... GPU array of ones.
% Compilation supported
% 
% EXAMPLE
% 
% A   =   ones(10,GPUsingle)
% B   =   ones(10, 10,GPUsingle)
% C   =   ones([10 10],GPUsingle)
% A   =   ones(10,GPUdouble)
% B   =   ones(10, 10,GPUdouble)
% C   =   ones([10 10],GPUdouble)
