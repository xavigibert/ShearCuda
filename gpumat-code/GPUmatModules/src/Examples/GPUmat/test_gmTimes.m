function test_gmTimes
GPUtestLOG('Testing test_gmTimes',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = times(A, B);
gmTimes(A, B, R);
compareCPUGPU(single(r), R);
end
