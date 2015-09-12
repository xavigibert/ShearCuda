function ex7
N = 10; 
A = GPUsingle(rand(1,N));
Isamin = cublasIsamin(N, getPtr(A), 1);

