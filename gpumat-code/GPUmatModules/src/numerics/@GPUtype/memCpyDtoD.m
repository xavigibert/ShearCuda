% memCpyDtoD - Device-Device memory copy
% 
% SYNTAX
% 
% memCpyDtoD(R, X, index, count)
% R - GPUsingle, GPUdouble
% X - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% memCpyDtoD(R, X, index, count) copies count elements from X
% to R(index)
% Compilation supported
% 
% EXAMPLE
% 
% R = rand(100,100,GPUsingle);
% X = rand(100,100,GPUsingle);
% memCpyDtoD(R, X, 100, 20)
