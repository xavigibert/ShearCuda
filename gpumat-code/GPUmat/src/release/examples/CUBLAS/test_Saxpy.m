function test_Saxpy
N = 10;
A = GPUsingle(rand(1,N));
B = GPUsingle(rand(1,N));

alpha = 2.0;
Saxpy_mat = alpha * single(A) + single(B);

cublasSaxpy(N, alpha, getPtr(A), 1, getPtr(B), 1);

status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');




end