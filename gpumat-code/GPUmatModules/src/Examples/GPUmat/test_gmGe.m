function test_gmGe
GPUtestLOG('Testing test_gmGe',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = ge(A, B);
gmGe(A, B, R);
compareCPUGPU(single(r), R);
end
