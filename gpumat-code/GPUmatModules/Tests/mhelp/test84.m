function test84
X = rand(5,5,GPUsingle)+i*rand(5,5,GPUsingle);
R = fft2(X);
X = ifft2(R);
