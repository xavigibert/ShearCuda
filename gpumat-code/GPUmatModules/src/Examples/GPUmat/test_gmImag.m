function test_gmImag
GPUtestLOG('Testing test_gmImag',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = imag(A);
gmImag(A, R);
compareCPUGPU(single(r), R);
end
