% GPUacosh - Inverse hyperbolic cosine
% 
% SYNTAX
% 
% GPUacosh(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUacosh(X, R) is equivalent to ACOSH(X), but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle) + 1;
% R = zeros(size(X), GPUsingle);
% GPUacosh(X, R)
