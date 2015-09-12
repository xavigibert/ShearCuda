% GPUcosh - Hyperbolic cosine
% 
% SYNTAX
% 
% GPUcosh(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUcosh(X, R) is equivalent to COSH(X) , but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUcosh(X, R)
