% randn - GPU pseudorandom generator
% 
% SYNTAX
% 
% randn(N,GPUsingle)
% randn(M,N,GPUsingle)
% randn([M,N],GPUsingle)
% randn(M,N,P,...?,GPUsingle)
% randn([M N P ...],GPUsingle)
% 
% randn(N,GPUdouble)
% randn(M,N,GPUdouble)
% randn([M,N],GPUdouble)
% randn(M,N,P,...?,GPUdouble)
% randn([M N P ...],GPUdouble)
% 
% 
% MODULE NAME
% RAND
% 
% DESCRIPTION
% randn(N,GPUsingle) is an N-by-N GPU matrix of values generated
% with a pseudorandom generator (normal distribution).
% randn(M,N,GPUsingle) or randn([M,N],GPUsingle) is an M-by-N
% GPU matrix.
% randn(M,N,P,...,GPUsingle)           or          randn([M N P
% ...,GPUsingle]) is an M-by-N-by-P-by-... GPU array of
% single precision values.
% randn(M,N,P,...,GPUdouble)           or          randn([M N P
% ...,GPUdouble]) is an M-by-N-by-P-by-... GPU array of
% double precision values.
% Compilation supported
% 
% EXAMPLE
% 
% A   =   randn(10,GPUsingle)
% B   =   randn(10, 10,GPUsingle)
% C   =   randn([10 10],GPUsingle)
% A   =   randn(10,GPUdouble)
% B   =   randn(10, 10,GPUdouble)
% C   =   randn([10 10],GPUdouble)
