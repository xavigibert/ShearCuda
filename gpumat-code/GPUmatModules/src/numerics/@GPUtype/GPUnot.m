% GPUnot - Logical NOT
% 
% SYNTAX
% 
% GPUnot(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUnot(X, R) is equivalent to not(X), but the result is returned
% in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = GPUsingle([1 2 0 4]);
% R = zeros(size(A), GPUsingle);
% GPUnot(A, R);
