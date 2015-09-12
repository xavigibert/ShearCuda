% GPUreal - Real part of complex number
% 
% SYNTAX
% 
% GPUreal(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUreal(X, R) is equivalent to real(X), but result is returned in
% the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle) + sqrt(-1)*rand(10,GPUsingle);
% R = zeros(size(A), GPUsingle);
% GPUreal(A, R);
