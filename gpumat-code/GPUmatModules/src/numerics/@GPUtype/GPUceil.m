% GPUceil - Round towards plus infinity
% 
% SYNTAX
% 
% GPUceil(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUceil(X, R) is equivalent to CEIL(X), but result is returned in
% the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUceil(X, R)
