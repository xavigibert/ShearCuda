% and - Logical AND
% 
% SYNTAX
% 
% R   =   A & B
% R   =   and(A,B)
% A   -   GPUsingle, GPUdouble
% B   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% A & B performs a logical AND of arrays A and B and returns an
% array containing elements set to either logical 1 (TRUE) or logical
% 0 (FALSE).
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 3 0 4]);
% B = GPUsingle([0 1 10 2]);
% R = A & B;
% single(R)
