function test_Sasum
N = 10;
A = GPUsingle(rand(1,N));

Sasum = cublasSasum( N, getPtr(A), 1);
status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');

Sasum_mat = sum(single(A));


end