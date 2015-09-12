% GPUacos - Inverse cosine
% 
% SYNTAX
% 
% GPUacos(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUacos(X, R) is equivalent to ACOS(X), but result is returned in
% the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUacos(X, R)
