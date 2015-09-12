% times - Array multiply
% 
% SYNTAX
% 
% R   =   X .* Y
% R   =   times(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% X.*Y denotes element-by-element multiplication. X and Y must
% have the same dimensions unless one is a scalar. A scalar can be
% multiplied into anything.
% Compilation supported
% 
% EXAMPLE
% 
% A   =   rand(10,GPUsingle);
% B   =   rand(10,GPUsingle);
% R   =   A .* B
% A   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% B   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% R   =   A .* B
% A   =   rand(10,GPUdouble)+i*rand(10,GPUdouble);
% B   =   rand(10,GPUdouble)+i*rand(10,GPUdouble);
% R   =   A .* B
