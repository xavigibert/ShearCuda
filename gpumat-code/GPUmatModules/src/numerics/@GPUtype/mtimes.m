% mtimes - Matrix multiply
% 
% SYNTAX
% 
% R   =   X * Y
% R   =   mtimes(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% * (mtimes(X, Y)) is the matrix product of X and Y.
% Compilation supported
% 
% EXAMPLE
% 
% A   =   rand(10,GPUsingle);
% B   =   rand(10,GPUsingle);
% R   =   A * B
% A   =   rand(10,GPUdouble);
% B   =   rand(10,GPUdouble);
% R   =   A * B
% A   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% B   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% R   =   A * B
