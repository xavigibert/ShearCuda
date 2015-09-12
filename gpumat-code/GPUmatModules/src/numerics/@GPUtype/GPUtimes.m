% GPUtimes - Array multiply
% 
% SYNTAX
% 
% GPUtimes(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUtimes(X, Y, R) is equivalent to times(X, Y), but the result
% is returned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% B = rand(10,GPUsingle);
% R = zeros(size(A), GPUsingle);
% GPUtimes(A, B, R);
