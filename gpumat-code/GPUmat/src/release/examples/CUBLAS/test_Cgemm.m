function test_Cgemm
N = 10;
I = sqrt(-1);
A = GPUsingle(rand(N,N) + I*rand(N,N));
B = GPUsingle(rand(N,N) + I*rand(N,N));

% C needs to be complex as well, thats why we multiply by I
C = zeros(N,N,GPUsingle)*I;

% alpha is complex
alpha = 2.0+I*3.0;
beta  = 0.0;

opA = 'n'; 
opB = 'n';
  
cublasCgemm(opA, opB, N, N, N, ... 
  alpha, getPtr(A), N, getPtr(B), ...
  N, beta, getPtr(C), N);

status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');

C_mat = alpha * single(A)*single(B);

end

