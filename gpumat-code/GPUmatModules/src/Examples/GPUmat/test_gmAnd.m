function test_gmAnd
GPUtestLOG('Testing test_gmAnd',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = and(A, B);
gmAnd(A, B, R);
compareCPUGPU(single(r), R);
end
