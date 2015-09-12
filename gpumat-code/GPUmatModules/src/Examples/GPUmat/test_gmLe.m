function test_gmLe
GPUtestLOG('Testing test_gmLe',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = le(A, B);
gmLe(A, B, R);
compareCPUGPU(single(r), R);
end
