% GPUuminus - Unary minus
% 
% SYNTAX
% 
% GPUuminus(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUuminus(X, R) is equivalent to uminus(X), but the result is re-
% turned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUuminus(X, R)
