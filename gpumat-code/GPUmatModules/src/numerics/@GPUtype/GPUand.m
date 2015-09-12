% GPUand - Logical AND
% 
% SYNTAX
% 
% GPUand(A, B, R)
% A - GPUsingle, GPUdouble
% B - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUand(A, B, R) is equivalent to A & B, but result is returned in
% the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 3 0 4]);
% B = GPUsingle([0 1 10 2]);
% R = zeros(size(A), GPUsingle);
% GPUand(A, B, R);
