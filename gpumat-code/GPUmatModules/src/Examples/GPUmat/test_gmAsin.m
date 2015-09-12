function test_gmAsin
GPUtestLOG('Testing test_gmAsin',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = asin(A);
gmAsin(A, R);
compareCPUGPU(single(r), R);
end
