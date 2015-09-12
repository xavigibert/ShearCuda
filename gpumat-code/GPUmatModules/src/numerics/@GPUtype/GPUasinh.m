% GPUasinh - Inverse hyperbolic sine
% 
% SYNTAX
% 
% GPUasinh(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUasinh(X, R) is equivalent to ASINH(X) , but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUasinh(X, R)
