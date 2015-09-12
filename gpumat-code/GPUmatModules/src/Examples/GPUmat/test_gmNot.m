function test_gmNot
GPUtestLOG('Testing test_gmNot',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = not(A);
gmNot(A, R);
compareCPUGPU(single(r), R);
end
