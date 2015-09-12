% GPUcomplex - Construct complex data from real and imaginary
% components
% 
% SYNTAX
% 
% GPUcomplex(X, R)
% GPUcomplex(X,Y,R)
% X - GPUsingle, GPUdouble
% Y - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUcomplex(X, R) is equivalent to complex(X), but result is re-
% turned in the input parameter R.
% Compilation supported
% 
% EXAMPLE
% 
% RE = rand(10,GPUsingle);
% IM = rand(10,GPUsingle);
% R = complex(zeros(size(RE), GPUsingle));
% GPUcomplex(RE, R);
% R = complex(RE, IM);
