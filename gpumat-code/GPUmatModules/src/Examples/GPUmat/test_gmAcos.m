function test_gmAcos
GPUtestLOG('Testing test_gmAcos',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = acos(A);
gmAcos(A, R);
compareCPUGPU(single(r), R);
end
