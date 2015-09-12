% fft - Discrete Fourier transform
% 
% SYNTAX
% 
% R = fft(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% FFT(X) is the discrete Fourier transform (DFT) of vector X.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
% R = fft(X)
