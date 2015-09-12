% asin - Inverse sine
% 
% SYNTAX
% 
% R = asin(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ASIN(X) is the arcsine of the elements of X. NaN (Not A Number)
% results are obtained if ABS(x) > 1.0 for some element.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(10,GPUsingle);
% R = asin(X)
% 
% 
% MATLAB COMPATIBILITY
% NaN returned if ABS(x) > 1.0 . In this case Matlab returns a
% complex number. Not implemented for complex X.
