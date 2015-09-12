% GPUle - Less than or equal
% 
% SYNTAX
% 
% GPUle(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUle(A, B, R) is equivalent to le(A, B), but result is returned
% in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% B = GPUsingle([1 0 0 4]);
% R = zeros(size(A), GPUsingle);
% GPUle(A, B, R);
