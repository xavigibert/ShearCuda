% GPUlog10 - Common (base 10) logarithm
% 
% SYNTAX
% 
% GPUlog10(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUlog10(X, R) is equivalent to LOG10(X), but the result is re-
% turned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUlog10(X, R)
