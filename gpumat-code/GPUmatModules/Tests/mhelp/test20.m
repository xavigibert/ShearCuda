function test20
X = rand(1,5,GPUsingle)+i*rand(1,5,GPUsingle);
R = fft(X)
