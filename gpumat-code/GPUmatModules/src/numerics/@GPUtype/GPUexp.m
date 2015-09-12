% GPUexp - Exponential
% 
% SYNTAX
% 
% GPUexp(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUexp(X, R) is equivalent to EXP(X), but result is returned in the
% input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
% R = complex(zeros(size(X), GPUsingle));
% GPUexp(X, R)
