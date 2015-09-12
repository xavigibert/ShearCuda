function ex3
A = GPUsingle([1 2; 3 4]);
% Ah should be single precision, because
% A is single precision
Ah = single(zeros(1,numel(A)));
[status Ah] = cublasGetVector (numel(A), ...
              getSizeOf(A), getPtr(A), 1, Ah, 1);
cublasCheckStatus( status, ...
          'Unable to retrieve variable values from GPU.');

