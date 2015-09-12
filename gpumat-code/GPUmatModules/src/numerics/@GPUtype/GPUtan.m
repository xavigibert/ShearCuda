% GPUtan - Tangent of argument in radians
% 
% SYNTAX
% 
% GPUtan(X,R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUtan(X, R) is equivalent to tan(X), but the result is returned
% in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUtan(X,R)
