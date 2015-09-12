% isreal - True for real array
% 
% SYNTAX
% 
% R = isreal(X)
% X - GPU variable
% R - logical (0 or 1)
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ISREAL(X) returns 1 if X does not have an imaginary part and 0
% otherwise.
% Compilation not supported
% 
% EXAMPLE
% 
% A = rand(5,GPUsingle);
% isreal(A)
% A = rand(5,GPUsingle)+i*rand(5,GPUsingle);
% isreal(A)
