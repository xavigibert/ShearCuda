% GPUlog1p - Compute log(1+z) accurately
% 
% SYNTAX
% 
% GPUlog1p(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUlog1p(X, R) is equivalent to LOG1P(X), but the result is re-
% turned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUlog1p(X, R)
