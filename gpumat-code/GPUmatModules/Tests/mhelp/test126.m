function test126
Bh = single(rand(100));
B = GPUsingle(Bh);
Ah = Bh(1:end);
A = slice(B,[1,1,END]);
Ah = Bh(1:10,:);
A = slice(B,[1,1,10],':');
Ah = Bh([2 3 1],:);
A = slice(B,{[2 3 1]},':');
Ah = Bh([2 3 1],1);
A = slice(B,{[2 3 1]},1);
Ah = Bh(:,:);
A = slice(B,':',':');
