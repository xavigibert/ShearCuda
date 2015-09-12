function test_cublasSetVector

B = [1 2 3 4];

% Create empty GPU variable A
A = GPUsingle();
setSize(A, size(B));
GPUallocVector(A); 

status = cublasSetVector(numel(A), getSizeOf(A), B, 1, getPtr(A), 1);
cublasCheckStatus( status, 'Unable to retrieve variable values from GPU.');

disp(single(A));


end
