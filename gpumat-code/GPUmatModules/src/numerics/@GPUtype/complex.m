% complex - Construct complex data from real and imaginary com-
% ponents
% 
% SYNTAX
% 
% R   =   complex(X)
% R   =   complex(X,Y)
% X   -   GPUsingle, GPUdouble
% Y   -   GPUsingle, GPUdouble
% R   -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% R = complex(X, Y) creates a complex output R from the two real
% inputs X and Y. R = complex(X) creates a complex output R from
% the real input X. Imaginary part is set to 0.
% Compilation supported
% 
% EXAMPLE
% 
% RE = rand(10,GPUsingle);
% IM = rand(10,GPUsingle);
% R = complex(RE);
% R = complex(RE, IM);
