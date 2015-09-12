% rdivide - Right array divide
% 
% SYNTAX
% 
% R   =   X ./ Y
% R   =   rdivide(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% A./B denotes element-by-element division. A and B must have the
% same dimensions unless one is a scalar. A scalar can be divided
% with anything.
% Compilation supported
% 
% EXAMPLE
% 
% A   =   rand(10,GPUsingle);
% B   =   rand(10,GPUsingle);
% R   =   A ./ B
% A   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% B   =   rand(10,GPUsingle)+i*rand(10,GPUsingle);
% R   =   A ./ B
