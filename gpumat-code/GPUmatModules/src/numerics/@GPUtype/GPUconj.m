% GPUconj - GPUconj(X, R) is the complex conjugate of X
% 
% SYNTAX
% 
% GPUconj(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUconj(X, R) is equivalent to CONJ(X), but result is returned in
% the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(1,5,GPUsingle) + i*rand(1,5,GPUsingle);
% R = complex(zeros(size(A), GPUsingle));
% GPUconj(A, R)
