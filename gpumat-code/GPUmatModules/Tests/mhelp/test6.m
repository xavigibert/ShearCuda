function test6
A = rand(100,GPUsingle);
B = rand(10,10,GPUsingle);
Ah = single(A);
Bh = single(B);
Ah(1:10,1:10) = Bh;
assign(1, A, B, [1,1,10],[1,1,10]);
assign(1, A, Bh, [1,1,10],[1,1,10]);
assign(1, A, single(10), [1,1,10],[1,1,10]);
