% GPUasin - Inverse sine
% 
% SYNTAX
% 
% GPUasin(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUasin(X, R) is equivalent to ASIN(X), but result is returned in
% input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUasin(X, R);
