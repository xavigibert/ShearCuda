% or - Logical OR
% 
% SYNTAX
% 
% R   =   X | Y
% R   =   or(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% A | B (or(A, B)) performs a logical OR of arrays A and B.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = A | B;
% single(R)
% R = or(A, B);
% single(R)
