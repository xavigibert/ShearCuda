% GPUatanh - Inverse hyperbolic tangent
% 
% SYNTAX
% 
% GPUatanh(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUatanh(X, R) is equivalent to ATANH(X), but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUatanh(X, R)
