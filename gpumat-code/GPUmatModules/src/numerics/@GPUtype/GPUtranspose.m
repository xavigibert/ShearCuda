% GPUtranspose - Transpose
% 
% SYNTAX
% 
% GPUtranspose(X, R)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUtranspose(X, R) is equivalent to transpose(X), but the result
% is returned in input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = zeros(size(X), GPUsingle);
% GPUtranspose(X, R)
