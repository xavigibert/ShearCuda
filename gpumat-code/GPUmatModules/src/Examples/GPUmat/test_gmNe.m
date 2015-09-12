function test_gmNe
GPUtestLOG('Testing test_gmNe',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = ne(A, B);
gmNe(A, B, R);
compareCPUGPU(single(r), R);
end
