% GPUrdivide - Right array divide
% 
% SYNTAX
% 
% GPUrdivide(X,Y)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUrdivide(X, Y, R) is equivalent to rdivide(X, Y), but the re-
% sult is returned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% B = rand(10,GPUsingle);
% R = zeros(size(A), GPUsingle);
% GPUrdivide(A, B, R);
