% GPUcompileStart - Starts the GPUmat compiler.
% 
% SYNTAX
% 
% GPUcompileStart(NAME, OPTIONS, X1, X2, ..., XN)
% NAME - Function name
% OPTIONS - Compilation options
% X1, X2, ..., XN - GPUsingle, GPUdouble, Matlab variables
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% Starts the GPUmat compiler. Check the manual for more informa-
% tion.
% 
% EXAMPLE
% 
% A = randn(5,GPUsingle); % A is a dummy variable
% % Compile function C=myexp(B)
% GPUcompileStart('myexp','-f',A)
% R = exp(A);
% GPUcompileStop(R)
