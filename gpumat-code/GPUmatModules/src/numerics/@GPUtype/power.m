% power - Array power
% 
% SYNTAX
% 
% R   =   X .^ Y
% R   =   power(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% Z = X.^Y denotes element-by-element powers.
% Compilation supported
% 
% EXAMPLE
% 
% A   =   rand(10,GPUsingle);
% B   =   2;
% R   =   A .^ B
% A   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% R   =   A .^ B
% 
% 
% MATLAB COMPATIBILITY
% Implemented for REAL exponents only.
