% acosh - Inverse hyperbolic cosine
% 
% SYNTAX
% 
% R = acosh(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ACOSH(X) is the inverse hyperbolic cosine of the elements of X.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle) + 1;
% R = acosh(X)
% 
% 
% MATLAB COMPATIBILITY
% NaN is returned if X<1.0 . Not implemented for complex X.
