% conj - CONJ(X) is the complex conjugate of X
% 
% SYNTAX
% 
% R = conj(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% For a complex X, CONJ(X) = REAL(X) - i*IMAG(X).
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(1,5,GPUsingle) + i*rand(1,5,GPUsingle);
% B = conj(A)
