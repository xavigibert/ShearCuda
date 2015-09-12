function test_gmFloor
GPUtestLOG('Testing test_gmFloor',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = floor(A);
gmFloor(A, R);
compareCPUGPU(single(r), R);
end
