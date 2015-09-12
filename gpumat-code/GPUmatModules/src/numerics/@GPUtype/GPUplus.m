% GPUplus - Plus
% 
% SYNTAX
% 
% GPUplus(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUplus(X, Y, R) is equivalent to plus(X, Y), but the result is
% returned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% B = rand(10,GPUsingle);
% R = zeros(size(B), GPUsingle);
% GPUplus(A, B, R);
