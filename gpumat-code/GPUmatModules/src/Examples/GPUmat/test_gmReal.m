function test_gmReal
GPUtestLOG('Testing test_gmReal',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = real(A);
gmReal(A, R);
compareCPUGPU(single(r), R);
end
