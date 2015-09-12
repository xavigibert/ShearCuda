% GPUlog - Natural logarithm
% 
% SYNTAX
% 
% GPUlog(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUlog(X,R) is equivalent to LOG(X), but the result is returned in
% input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUlog(X,R)
