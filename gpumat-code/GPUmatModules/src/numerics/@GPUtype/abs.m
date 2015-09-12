% abs - Absolute value
% 
% SYNTAX
% 
% R = abs(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ABS(X) is the absolute value of the elements of X. When X is com-
% plex, ABS(X) is the complex modulus (magnitude) of the elements
% of X.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
% R = abs(X)
