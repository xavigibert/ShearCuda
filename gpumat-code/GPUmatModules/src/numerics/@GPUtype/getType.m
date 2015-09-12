% getType - Get the type of the GPU variable
% 
% SYNTAX
% 
% R = getType(X)
% X - GPU variable
% R - the type of the GPU variable
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% This is a low level function used to get the type of the GPU variable
% (FLOAT = 0, COMPLEX FLOAT = 1, DOUBLE = 2, COMPLEX
% DOUBLE = 3)
% Compilation not supported
% 
% EXAMPLE
% 
% A = rand(10,GPUsingle);
% getType(A)
