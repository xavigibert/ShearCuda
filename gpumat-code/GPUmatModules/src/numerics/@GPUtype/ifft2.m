% ifft2 - Two-dimensional inverse discrete Fourier transform
% 
% SYNTAX
% 
% R = ifft2(X)
% X - GPUsingle, GPUdouble
% R - GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% IFFT2(F) returns the two-dimensional inverse Fourier transform of
% matrix F.
% Compilation supported
% 
% EXAMPLE
% 
% X = rand(5,5,GPUsingle)+i*rand(5,5,GPUsingle);
% R = fft2(X);
% X = ifft2(R);
