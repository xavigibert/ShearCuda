% GPUabs - Absolute value
% 
% SYNTAX
% 
% R = GPUabs(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUabs(X, R) is equivalent to ABS(X), but result is returned in the
% input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
% R = zeros(size(X),GPUsingle);
% GPUabs(X, R)
