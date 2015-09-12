% GPUmtimes - Matrix multiply
% 
% SYNTAX
% 
% GPUmtimes(X,Y,R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUmtimes(X, Y, R) is equivalent to mtimes(X, Y), but the result
% is returned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% B = rand(10,GPUsingle);
% R = zeros(size(A), GPUsingle);
% GPUmtimes(A, B, R);
