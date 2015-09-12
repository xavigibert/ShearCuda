% GPUgt - Greater than
% 
% SYNTAX
% 
% GPUgt(X,Y, R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUgt(A, B, R) is equivalent to gt(A, B), but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = zeros(size(B), GPUsingle);
% GPUgt(A, B, R);
