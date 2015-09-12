function test_gmMinus
GPUtestLOG('Testing test_gmMinus',0);
A = GPUsingle(rand(10));
B = GPUsingle(rand(10));
R = zeros(size(A),GPUsingle);
r = minus(A, B);
gmMinus(A, B, R);
compareCPUGPU(single(r), R);
end
