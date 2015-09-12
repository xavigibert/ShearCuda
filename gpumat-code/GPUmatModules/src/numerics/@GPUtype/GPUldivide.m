% GPUldivide - Left array divide
% 
% SYNTAX
% 
% GPUldivide(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUldivide(A, B, R) is equivalent to ldivide(A, B), but result
% is returned in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% B = rand(10,GPUsingle);
% R = zeros(size(B), GPUsingle);
% GPUldivide(A, B, R);
