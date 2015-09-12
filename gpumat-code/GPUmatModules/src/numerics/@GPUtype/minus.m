% minus - Minus
% 
% SYNTAX
% 
% R   =   X - Y
% R   =   minus(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% X - Y subtracts matrix Y from X. X and Y must have the same
% dimensions unless one is a scalar. A scalar can be subtracted from
% anything.
% Compilation supported
% 
% EXAMPLE
% 
% X   =   rand(10,GPUsingle);
% Y   =   rand(10,GPUsingle);
% R   =   Y - X
% X   =   rand(10,GPUdouble);
% Y   =   rand(10,GPUdouble);
% R   =   Y - X
