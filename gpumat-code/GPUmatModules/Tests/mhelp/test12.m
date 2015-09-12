function test12
RE = rand(10,GPUsingle);
IM = rand(10,GPUsingle);
R = complex(RE);
R = complex(RE, IM);
