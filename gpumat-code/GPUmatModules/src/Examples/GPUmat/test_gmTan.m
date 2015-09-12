function test_gmTan
GPUtestLOG('Testing test_gmTan',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = tan(A);
gmTan(A, R);
compareCPUGPU(single(r), R);
end
