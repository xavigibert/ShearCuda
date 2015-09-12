function test_gmExp
GPUtestLOG('Testing test_gmExp',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = exp(A);
gmExp(A, R);
compareCPUGPU(single(r), R);
end
