function test_gmPower
GPUtestLOG('Testing test_gmPower',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = power(A, B);
gmPower(A, B, R);
compareCPUGPU(single(r), R);
end
