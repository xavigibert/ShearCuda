% transpose - Transpose
% 
% SYNTAX
% 
% R   =   X.'
% R   =   transpose(X)
% X   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% X.' or transpose(X) is the non-conjugate transpose.
% Compilation supported
% 
% EXAMPLE
% 
% X   =   rand(10,GPUsingle);
% X   =   rand(10,GPUdouble);
% R   =   X.'
% R   =   transpose(X)
