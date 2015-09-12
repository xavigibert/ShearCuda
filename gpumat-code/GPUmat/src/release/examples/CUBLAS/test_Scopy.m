function test_Scopy
N = 10;
A = GPUsingle(rand(1,N));
B = GPUsingle(rand(1,N));

cublasScopy(N, getPtr(A), 1, getPtr(B), 1);
status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');



end