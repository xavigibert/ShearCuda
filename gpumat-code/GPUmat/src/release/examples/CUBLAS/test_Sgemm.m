function test_Sgemm
N = 10;
A = GPUsingle(rand(N,N));
B = GPUsingle(rand(N,N));
C = zeros(N,N,GPUsingle);

alpha = 2.0;
beta  = 0.0;

opA = 'n'; 
opB = 'n';
  
cublasSgemm(opA, opB, N, N, N, ... 
  alpha, getPtr(A), N, getPtr(B), ...
  N, beta, getPtr(C), N);

status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');

C_mat = alpha * single(A)*single(B);


end

