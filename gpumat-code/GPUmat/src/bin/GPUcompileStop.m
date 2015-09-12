% GPUcompileStop - Stops the GPUmat compiler.
% 
% SYNTAX
% 
% GPUcompileStop(X1, X2, ..., XN)
% X1, X2, ..., XN - GPUsingle, GPUdouble, Matlab variables
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% Stops the GPUmat compiler. Check the manual for more informa-
% tion.
% 
% EXAMPLE
% 
% A = randn(5,GPUsingle); % A is a dummy variable
% % Compile function C=myexp(B)
% GPUcompileStart('myexp','-f',A)
% R = exp(A);
% GPUcompileStop(R)
