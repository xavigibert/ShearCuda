% iscomplex - True for complex array
% 
% SYNTAX
% 
% R = iscomplex(X)
% X - GPU variable
% R - logical (0 or 1)
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ISCOMPLEX(X) returns 1 if X does have an imaginary part and 0
% otherwise.
% Compilation not supported
% 
% EXAMPLE
% 
% A = rand(5,GPUsingle);
% iscomplex(A)
% A = rand(5,GPUsingle)+i*rand(5,GPUsingle);
% iscomplex(A)
