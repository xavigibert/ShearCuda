function test98
R = rand(100,100,GPUsingle);
X = rand(100,100,GPUsingle);
memCpyDtoD(R, X, 100, 20)
