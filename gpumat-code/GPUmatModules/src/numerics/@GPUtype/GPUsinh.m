% GPUsinh - Hyperbolic sine
% 
% SYNTAX
% 
% GPUsinh(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUsinh(X, R) is equivalent to sinh(X), but the result is returned
% in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUsinh(X,R)
