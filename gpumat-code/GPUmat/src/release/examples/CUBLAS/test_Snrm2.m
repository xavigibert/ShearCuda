function test_Snrm2
N = 10;
A = GPUsingle(rand(1,N));

Snrm2_mat = sqrt(sum(single(A).*single(A)));
Snrm2 = cublasSnrm2(N, getPtr(A),1);

status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');



end