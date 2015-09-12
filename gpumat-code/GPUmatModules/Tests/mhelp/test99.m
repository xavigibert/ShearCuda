function test99
R = rand(100,100,GPUsingle);
X = single(rand(100,100));
memCpyHtoD(R, X, 100, 20)
