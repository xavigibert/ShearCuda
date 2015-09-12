% GPUlog2 - Base 2 logarithm and dissect floating point number
% 
% SYNTAX
% 
% GPUlog2(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUlog2(X, R) is equivalent to LOG2(X), but the result is returned
% in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUlog2(X, R)
