% GPUimag - Imaginary part of complex number
% 
% SYNTAX
% 
% GPUimag(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUimag(X, R) is equivalent to imag(X), but result is returned in
% the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle) + sqrt(-1)*rand(10,GPUsingle);
% R = zeros(size(A), GPUsingle);
% GPUimag(A, R);
