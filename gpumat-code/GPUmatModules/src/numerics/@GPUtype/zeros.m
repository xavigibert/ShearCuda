% zeros - GPU zeros array
% 
% SYNTAX
% 
% zeros(N,GPUsingle)
% zeros(M,N,GPUsingle)
% zeros([M,N],GPUsingle)
% zeros(M,N,P,...?,GPUsingle)
% zeros([M N P ...],GPUsingle)
% 
% zeros(N,GPUdouble)
% zeros(M,N,GPUdouble)
% zeros([M,N],GPUdouble)
% zeros(M,N,P,...?,GPUdouble)
% zeros([M N P ...],GPUdouble)
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% zeros(N,GPUsingle) is an N-by-N GPU matrix of zeros.
% zeros(M,N,GPUsingle) or zeros([M,N],GPUsingle) is an M-by-N
% GPU matrix of single precision zeros.
% zeros(M,N,P,...,GPUsingle)            or      zeros([M N P
% ...,GPUsingle]) is an M-by-N-by-P-by-... GPU array of
% single precision zeros.
% zeros(M,N,P,...,GPUdouble)            or      zeros([M N P
% ...,GPUdouble]) is an M-by-N-by-P-by-... GPU array of
% double precision zeros.
% Compilation supported
% 
% EXAMPLE
% 
% A = zeros(10,GPUsingle)
% B = zeros(10, 10,GPUsingle)
% C = zeros([10 10],GPUsingle)
% 
% A = zeros(10,GPUdouble)
% B = zeros(10, 10,GPUdouble)
% C = zeros([10 10],GPUdouble)
