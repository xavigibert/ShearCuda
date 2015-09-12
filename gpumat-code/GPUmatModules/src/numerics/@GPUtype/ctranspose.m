% ctranspose - Complex conjugate transpose
% 
% SYNTAX
% 
% R   =   X'
% R   =   ctranspose(X)
% X   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% X' is the complex conjugate transpose of X.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle)+i*rand(10,GPUsingle);
% R = X'
% R = ctranspose(X)
