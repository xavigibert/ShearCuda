function simpleCUBLAS
% This is the GPUmat translation of the code in the CUDA SDK projects called
% with the same name (simpleCUBLAS). 
% The example show how to access CUBLAS functions from GPUmat

SIZEOF_FLOAT = sizeoffloat();

%% Allocate HOST arrays and initialize with random numbers
N = 500;

h_A = single(rand(N));
h_B = single(rand(N));
h_C = single(rand(N));


%% Allocate GPU arrays
d_A = GPUsingle(h_A);
d_B = GPUsingle(h_B);
d_C = GPUsingle(h_C);

% Although d_A was already initialized with h_A values, we can
% call cublasSetVector to do that again
status = cublasSetVector(N*N, SIZEOF_FLOAT, ...
                         h_A, 1, getPtr(d_A), 1);
cublasCheckStatus( status, '!!!! device access error (write A)');

% Calculate reference in Matlab
alpha = 2.0;
h_C_ref = alpha * h_A*h_B;

% Execute on GPU
cublasSgemm('n','n', N, N, N, alpha, getPtr(d_A), ...
            N, getPtr(d_B), N, 0.0, getPtr(d_C), N);
status = cublasGetError();
cublasCheckStatus( status, '!!!! kernel execution error.');

% Copy results back to HOST
h_C = single(d_C);


% Clean up GPU memory
% THERE IS NO NEED TO CLEAN UP MEMORY
% NEVERTHELESS, IF NECESSARY, ALWAYS USE
% CLEAR WITH GPUSINGLE
clear d_A
clear d_B
clear d_C

end
