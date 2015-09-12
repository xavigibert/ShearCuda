function test_gmConj
GPUtestLOG('Testing test_gmConj',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = conj(A);
gmConj(A, R);
compareCPUGPU(single(r), R);
end
