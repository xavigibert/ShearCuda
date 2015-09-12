% isempty - True for empty GPUsingle array
% 
% SYNTAX
% 
% R = isempty(X)
% X - GPU variable
% R - logical (0 or 1)
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ISEMPTY(X) returns 1 if X is an empty GPUsingle array and 0
% otherwise. An empty GPUsingle array has no elements, that is
% prod(size(X))==0.
% Compilation not supported
% 
% EXAMPLE
% 
% A = GPUsingle();
% isempty(A)
% A = rand(5,GPUsingle)+i*rand(5,GPUsingle);
% isempty(A)
