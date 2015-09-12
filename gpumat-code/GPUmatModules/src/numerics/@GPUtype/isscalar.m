% isscalar - True if array is a scalar
% 
% SYNTAX
% 
% R = isscalar(X)
% X - GPU variable
% R - logical (0 or 1)
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ISSCALAR(S) returns 1 if S is a 1x1 matrix and 0 otherwise.
% Compilation not supported
% 
% EXAMPLE
% 
% A = rand(5,GPUsingle);
% isscalar(A)
% A = GPUsingle(1);
% isscalar(A)
% A = GPUdouble(1);
% isscalar(A)
