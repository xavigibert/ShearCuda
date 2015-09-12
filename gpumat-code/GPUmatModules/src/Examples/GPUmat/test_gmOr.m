function test_gmOr
GPUtestLOG('Testing test_gmOr',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = or(A, B);
gmOr(A, B, R);
compareCPUGPU(single(r), R);
end
