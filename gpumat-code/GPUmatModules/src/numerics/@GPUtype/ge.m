% ge - Greater than or equal
% 
% SYNTAX
% 
% R   =   X >= Y
% R   =   ge(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% A >= B (ge(A, B)) does element by element comparisons between
% A and B.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = A >= B;
% single(R)
% R = ge(A, B);
% single(R)
