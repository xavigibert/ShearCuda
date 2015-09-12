function test_gmEq
GPUtestLOG('Testing test_gmEq',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = eq(A, B);
gmEq(A, B, R);
compareCPUGPU(single(r), R);
end
