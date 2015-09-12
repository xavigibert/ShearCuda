% GPUor - Logical OR
% 
% SYNTAX
% 
% GPUor(X,Y, R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUor(X, Y, R) is equivalent to or(X, Y), but the result is re-
% turned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = zeros(size(B), GPUsingle);
% GPUor(A, B, R);
