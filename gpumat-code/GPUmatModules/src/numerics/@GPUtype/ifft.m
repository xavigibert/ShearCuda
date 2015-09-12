% ifft - Inverse discrete Fourier transform
% 
% SYNTAX
% 
% R = ifft(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% IFFT(X) is the inverse discrete Fourier transform of X.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
% R = fft(X);
% X = ifft(R);
