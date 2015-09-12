% size - Size of array
% 
% SYNTAX
% 
% R = size(X)
% [M,N] = SIZE(X)
% [M1,M2,...,MN] = SIZE(X)
% X - GPU variable
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% D = SIZE(X), for M-by-N matrix X, returns the two-element row
% vector D = [M,N] containing the number of rows and columns in
% the matrix.
% Compilation not supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% size(X)
% X = rand(10,GPUdouble);
% size(X)
