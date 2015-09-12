% memCpyHtoD - Host-Device memory copy
% 
% SYNTAX
% 
% memCpyHtoD(R, X, index, count)
% R - GPUsingle, GPUdouble
% X - Matlab array
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% memCpyHtoD(R, X, index, count) copies count elements from
% the Matlab variable X (CPU) to R(index)
% Compilation supported
% 
% EXAMPLE
% 
% R = rand(100,100,GPUsingle);
% X = single(rand(100,100));
% memCpyHtoD(R, X, 100, 20)
