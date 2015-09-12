% GPUcos - Cosine of argument in radians
% 
% SYNTAX
% 
% GPUcos(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUcos(X, R) is equivalent to COS(X), but result is returned in the
% input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUcos(X, R)
