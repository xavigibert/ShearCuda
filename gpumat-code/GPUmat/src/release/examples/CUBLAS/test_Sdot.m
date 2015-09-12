function test_Sdot
N = 10;
A = GPUsingle(rand(1,N));
B = GPUsingle(rand(1,N));

Sdot_mat = sum(single(A).*single(B));
Sdot = cublasSdot(N, getPtr(A), 1, getPtr(B), 1);

status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');



end