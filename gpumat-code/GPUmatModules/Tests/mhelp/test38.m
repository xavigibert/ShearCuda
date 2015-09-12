function test38
A = rand(1,5,GPUsingle) + i*rand(1,5,GPUsingle);
R = complex(zeros(size(A), GPUsingle));
GPUconj(A, R)
