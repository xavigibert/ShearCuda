% repmat - Replicate and tile an array
% 
% SYNTAX
% 
% R   =   repmat(X,M,N)
% R   =   REPMAT(X,[M N])
% R   =   REPMAT(X,[M N P ...])
% R   -   GPUsingle, GPUdouble
% X   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% R = repmat(X,M,N) creates a large matrix R consisting of an
% M-by-N tiling of copies of X. The statement repmat(X,N) creates
% an N-by-N tiling.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% repmat(A,3,4,5)
