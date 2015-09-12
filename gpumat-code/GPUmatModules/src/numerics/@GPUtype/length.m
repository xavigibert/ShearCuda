% length - Length of vector
% 
% SYNTAX
% 
% R = length(X)
% X - GPU variable
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% LENGTH(X) returns the length of vector X. It is equivalent to
% MAX(SIZE(X)) for non-empty arrays and 0 for empty ones.
% Compilation not supported
% 
% EXAMPLE
% 
% A = rand(5,GPUsingle);
% length(A)
