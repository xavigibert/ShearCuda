% GPUminus - Minus
% 
% SYNTAX
% 
% GPUminus(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUminus(X, Y, R) is equivalent to minus(X, Y), but the result
% is returned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% Y = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUminus(Y, X, R);
