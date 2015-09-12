% fft2 - Two-dimensional discrete Fourier Transform
% 
% SYNTAX
% 
% R = fft2(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% FFT2(X) returns the two-dimensional Fourier transform of matrix
% X.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(5,5,GPUsingle)+i*rand(5,5,GPUsingle);
% R = fft2(X)
