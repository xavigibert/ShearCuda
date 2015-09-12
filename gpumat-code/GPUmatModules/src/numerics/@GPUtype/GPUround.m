% GPUround - Round towards nearest integer
% 
% SYNTAX
% 
% GPUround(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUround(X, R) is equivalent to round(X), but the result is re-
% turned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUround(X,R);
