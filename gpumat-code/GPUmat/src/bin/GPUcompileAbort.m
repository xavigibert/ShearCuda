% GPUcompileAbort - Aborts the GPUmat compilation.
% 
% SYNTAX
% 
% GPUcompileAbort
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% Aborts the GPUmat compilation. Check the manual for more in-
% formation.
% 
% EXAMPLE
% 
% A = randn(5,GPUsingle); % A is a dummy variable
% % Compile function C=myexp(B)
% GPUcompileStart('myexp','-f',A)
% R = exp(A);
% GPUcompileAbort
