% numel - Number of elements in an array or subscripted array ex-
% pression.
% 
% SYNTAX
% 
% R = numel(X)
% X - GPU variable
% R - number of elements
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% N = NUMEL(A) returns the number of elements N in array A.
% Compilation not supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% numel(X)
% X = rand(10,GPUdouble);
% numel(X)
