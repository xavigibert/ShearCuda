% log1p - Compute log(1+z) accurately
% 
% SYNTAX
% 
% R = log1p(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% LOG1P(Z) computes log(1+z). Only REAL values are accepted.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = log1p(X)
% 
% 
% MATLAB COMPATIBILITY
% Not implemented for complex X.
