% real - Real part of complex number
% 
% SYNTAX
% 
% R = real(X)
% X - GPUsingle, GPUdouble
% 
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% R = real(X) returns the real part of the elements of X.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle) + sqrt(-1)*rand(10,GPUsingle);
% R = real(A);
