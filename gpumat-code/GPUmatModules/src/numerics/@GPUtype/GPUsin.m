% GPUsin - Sine of argument in radians
% 
% SYNTAX
% 
% GPUsin(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUsin(X, R) is equivalent to sin(X), but the result is returned
% in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUsin(X,R)
