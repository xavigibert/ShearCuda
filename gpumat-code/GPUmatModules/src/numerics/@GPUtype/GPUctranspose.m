% GPUctranspose - Complex conjugate transpose
% 
% SYNTAX
% 
% GPUctranspose(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUctranspose(X, R) is equivalent to ctranspose(X), but result
% is returned in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle)+i*rand(10,GPUsingle);
% R = complex(zeros(size(X), GPUsingle));
% GPUctranspose(X, R)
