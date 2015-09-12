% log10 - Common (base 10) logarithm
% 
% SYNTAX
% 
% R = log10(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% LOG10(X) is the base 10 logarithm of the elements of X. NaN results
% are produced if X is not positive.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = log10(X)
% 
% 
% MATLAB COMPATIBILITY
% Not implemented for complex X.
