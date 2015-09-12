function test_cublasAlloc
N = 10;
SIZEOF_FLOAT = sizeoffloat();

% GPU variable d_A
d_A = 0;
[status d_A] = cublasAlloc(N,SIZEOF_FLOAT,d_A);
ret = cublasCheckStatus( status, ...
  '!!!! device memory allocation error (d_A)');

% Clean up memory
status = cublasFree(d_A);
ret = cublasCheckStatus( status, ...
  '!!!! memory free error (d_A)');


end
