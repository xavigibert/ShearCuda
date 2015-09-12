function test_gmLog
GPUtestLOG('Testing test_gmLog',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = log(A);
gmLog(A, R);
compareCPUGPU(single(r), R);
end
