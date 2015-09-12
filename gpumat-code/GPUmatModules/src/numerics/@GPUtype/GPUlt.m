% GPUlt - Less than
% 
% SYNTAX
% 
% GPUlt(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUlt(X, Y, R) is equivalent to lt(X, Y), but the result is re-
% turned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = zeros(size(B), GPUsingle);
% GPUlt(A, B, R);
