function ex8
A = GPUsingle(rand(5));
iscomplex(A)
A = GPUsingle(rand(5)+i*rand(5));
iscomplex(A)

