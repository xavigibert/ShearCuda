function test_cufftExecC2C
fftType = cufftType;
fftDir  = cufftTransformDirections;

I = sqrt(-1);

A = GPUsingle(rand(1,128)+I*rand(1,128));
plan = 0;
[status, plan] = cufftPlan1d(plan, numel(A),  fftType.CUFFT_C2C, 1);
cufftCheckStatus(status, 'Error in cufftPlan1D');

[status] = cufftExecC2C(plan, getPtr(A),getPtr(A), fftDir.CUFFT_FORWARD);
cufftCheckStatus(status, 'Error in cufftExecC2C');

[status] = cufftDestroy(plan);
cufftCheckStatus(status, 'Error in cuffDestroyPlan');


end