function test_gmTranspose
GPUtestLOG('Testing test_gmTranspose',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = transpose(A);
gmTranspose(A, R);
compareCPUGPU(single(r), R);
end
