function simpleCUFFT
%% CUFFT example

%% Allocate HOST arrays and initialize with random numbers
N = 512;

h_A = single(rand(1,N)+i*rand(1,N));

d_A = GPUsingle(h_A);
d_B = GPUsingle(h_A);

fftType = cufftType;
fftDir  = cufftTransformDirections;

% FFT plan
plan = 0;
[status, plan] = cufftPlan1d(plan, numel(d_A),  fftType.CUFFT_C2C, 1);
cufftCheckStatus(status, 'Error in cufftPlan1D');

% Run GPU FFT
[status] = cufftExecC2C(plan, getPtr(d_A), getPtr(d_B), fftDir.CUFFT_FORWARD);
cufftCheckStatus(status, 'Error in cufftExecC2C');

% Run GPU IFFT
[status] = cufftExecC2C(plan, getPtr(d_B), getPtr(d_A), fftDir.CUFFT_INVERSE);
cufftCheckStatus(status, 'Error in cufftExecC2C');

% results should be scaled by 1/N if compared to CPU
h_B = 1/N*single(d_A);


[status] = cufftDestroy(plan);
cufftCheckStatus(status, 'Error in cuffDestroyPlan');

