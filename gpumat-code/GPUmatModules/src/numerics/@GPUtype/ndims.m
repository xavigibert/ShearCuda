% ndims - Number of dimensions
% 
% SYNTAX
% 
% R = ndims(X)
% X - GPU variable
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% N = NDIMS(X) returns the number of dimensions in the array X.
% The number of dimensions in an array is always greater than or
% equal to 2. Trailing singleton dimensions are ignored. Put simply,
% it is LENGTH(SIZE(X)).
% Compilation not supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% ndims(X)
