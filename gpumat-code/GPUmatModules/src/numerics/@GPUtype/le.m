% le - Less than or equal
% 
% SYNTAX
% 
% R   =   X <= Y
% R   =   le(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% A <= B (le(A, B)) does element by element comparisons between
% A and B.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = A <= B;
% single(R)
% R = le(A, B);
% single(R)
