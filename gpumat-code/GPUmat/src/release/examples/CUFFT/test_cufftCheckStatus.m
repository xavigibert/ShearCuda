function test_cufftCheckStatus
fftType = cufftType;
I = sqrt(-1);
A = GPUsingle(rand(1,128)+I*rand(1,128));
plan = 0;
[status, plan] = cufftPlan1d(plan, numel(A),  fftType.CUFFT_C2C, 1);
cufftCheckStatus(status, 'Error in cufftPlan1D');
end