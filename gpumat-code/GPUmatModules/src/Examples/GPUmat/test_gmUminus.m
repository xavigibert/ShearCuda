function test_gmUminus
GPUtestLOG('Testing test_gmUminus',0);
A = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = uminus(A);
gmUminus(A, R);
compareCPUGPU(single(r), R);
end
