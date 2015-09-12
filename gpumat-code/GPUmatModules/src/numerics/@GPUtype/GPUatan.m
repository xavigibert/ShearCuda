% GPUatan - Inverse tangent, result in radians
% 
% SYNTAX
% 
% GPUatan(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUatan(X, R) is equivalent to ATAN(X), but result is returned in
% the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUatan(X, R)
