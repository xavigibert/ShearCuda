% plus - Plus
% 
% SYNTAX
% 
% R   =   X + Y
% R   =   plus(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% X + Y (plus(X, Y)) adds matrices X and Y. X and Y must have
% the same dimensions unless one is a scalar (a 1-by-1 matrix). A
% scalar can be added to anything.
% Compilation supported
% 
% EXAMPLE
% 
% A   =   rand(10,GPUsingle);
% B   =   rand(10,GPUsingle);
% R   =   A + B
% A   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% B   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% R   =   A + B
