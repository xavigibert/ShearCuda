function test_gmLog1p
GPUtestLOG('Testing test_gmLog1p',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = log1p(A);
gmLog1p(A, R);
compareCPUGPU(single(r), R);
end
