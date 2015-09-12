% GPUge - Greater than or equal
% 
% SYNTAX
% 
% GPUge(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUge(A, B, R) is equivalent to ge(A, B), but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = zeros(size(B),GPUsingle);
% GPUge(A, B, R);
