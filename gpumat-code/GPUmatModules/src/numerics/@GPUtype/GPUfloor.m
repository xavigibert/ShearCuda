% GPUfloor - Round towards minus infinity
% 
% SYNTAX
% 
% GPUfloor(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUfloor(X, R) is equivalent to FLOOR(X), but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(1,5,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUfloor(X, R)
