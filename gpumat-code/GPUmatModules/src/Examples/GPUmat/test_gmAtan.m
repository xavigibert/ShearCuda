function test_gmAtan
GPUtestLOG('Testing test_gmAtan',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = atan(A);
gmAtan(A, R);
compareCPUGPU(single(r), R);
end
