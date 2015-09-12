function test37
RE = rand(10,GPUsingle);
IM = rand(10,GPUsingle);
R = complex(zeros(size(RE), GPUsingle));
GPUcomplex(RE, R);
R = complex(RE, IM);
