function test_Isamin
N = 10;
A = GPUsingle(rand(1,N));

Isamin = cublasIsamin(N, getPtr(A), 1);
status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');

[value, Isamin_mat] = min(single(A));


end
