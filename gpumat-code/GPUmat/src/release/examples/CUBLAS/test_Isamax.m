function test_Isamax
%% Low level implementation

N = 10;
SIZEOF_FLOAT = sizeoffloat();

% HOST variable h_A
h_A = single(rand(1,N));

% GPU variable d_A
d_A = 0.0;
[status d_A]= cublasAlloc(N,SIZEOF_FLOAT,d_A);
ret = cublasCheckStatus( status, ...
  '!!!! device memory allocation error (d_A)');

% Transfer from h_A to d_A
status = cublasSetVector(N,SIZEOF_FLOAT,h_A,1,d_A,1);
ret = cublasCheckStatus( status, ...
  '!!!! device access error (write d_A)');

Isamax = cublasIsamax(N, d_A, 1);
status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');

% Compare with Matlab
[value, Isamax_mat] = max(h_A);

% Clean up memory
status = cublasFree(d_A);
ret = cublasCheckStatus( status, ...
  '!!!! memory free error (d_A)');

%% High level implementation (GPUsingle)
A = GPUsingle(rand(1,N));

Isamax = cublasIsamax(N, getPtr(A), 1);
status = cublasGetError();
ret = cublasCheckStatus( status, ...
  '!!!! kernel execution error.');

[value, Isamax_mat] = max(single(A));

end
