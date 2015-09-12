% imag - Imaginary part of complex number
% 
% SYNTAX
% 
% R = imag(X)
% X - GPUsingle, GPUdouble
% 
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% R = imag(X) returns the imaginary part of the elements of X.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle) + sqrt(-1)*rand(10,GPUsingle);
% R = imag(A);
