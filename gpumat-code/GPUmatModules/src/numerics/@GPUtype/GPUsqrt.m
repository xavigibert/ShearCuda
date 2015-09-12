% GPUsqrt - Square root
% 
% SYNTAX
% 
% GPUsqrt(X,R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUsqrt(X, R) is equivalent to sqrt(X), but the result is returned
% in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUsqrt(X,R)
