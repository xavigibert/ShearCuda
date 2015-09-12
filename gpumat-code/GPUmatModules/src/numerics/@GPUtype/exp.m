% exp - Exponential
% 
% SYNTAX
% 
% R = exp(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% EXP(X) is the exponential of the elements of X, e to the X. For
% complex Z=X+i*Y, EXP(Z) = EXP(X)*(COS(Y)+i*SIN(Y)).
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
% R = exp(X)
