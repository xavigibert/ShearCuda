function test_gmRound
GPUtestLOG('Testing test_gmRound',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = round(A);
gmRound(A, R);
compareCPUGPU(single(r), R);
end
