function ComplexMatrixMultiplication
%% Complex Matrix-Matrix multiplication 

%% Allocate HOST arrays and initialize with random numbers
N = 500;

h_A = single(rand(N)+i*rand(N));
h_B = single(rand(N)+i*rand(N));
h_C = single(rand(N)+i*rand(N));

%% Allocate GPU arrays
d_A = GPUsingle(h_A);
d_B = GPUsingle(h_B);
d_C = GPUsingle(h_C);


% Calculate reference in Matlab
alpha = 2.0+3.0*i;
beta = 0.0;

h_C_ref = h_A*h_B;
h_C_ref = alpha * h_C_ref; 

% Execute on GPU
cublasCgemm('n','n', N, N, N, alpha, getPtr(d_A), ...
            N, getPtr(d_B), N, beta, getPtr(d_C), N);
status = cublasGetError();
cublasCheckStatus( status, '!!!! kernel execution error.');

% Copy results back to HOST
h_C = single(d_C);

% Clean up GPU memory
% THERE IS NO NEED TO CLEAN UP MEMORY
% NEVERTHELESS, IF NECESSARY, CLEAR
% A GPUSINGLE USING CLEAR
clear d_A
clear d_B
clear d_C

end
