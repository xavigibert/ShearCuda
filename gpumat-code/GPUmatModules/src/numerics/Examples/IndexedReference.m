function IndexedReference
%% Examples from Wiki page
%% http://sourceforge.net/apps/mediawiki/gpumatmodules/index.php?title=Indexed_references

B = GPUsingle(rand(100));
Bh = single(B);

%%
Ah = Bh(1:10);
A = zeros(size(Ah),GPUsingle);
ex10a(A,B);
compareCPUGPU(Ah,A);

%%
Ah = Bh(1:10);
A = zeros(size(Ah),GPUsingle);
ex10b(A,B);
compareCPUGPU(Ah,A);

%%
Ah = Bh(1:end);
A = zeros(size(Ah),GPUsingle);
ex10c(A,B);
compareCPUGPU(Ah,A);

%%
Ah = Bh(1:10,1:end);
A = zeros(size(Ah),GPUsingle);
ex10d(A,B);
compareCPUGPU(Ah,A);

%%
Ah = Bh([3 4 6 1]);
A = zeros(size(Ah),GPUsingle);
ex10e(A,B);
compareCPUGPU(Ah,A);

%%
Ah = Bh([3 4 6 1],1:end);
A = zeros(size(Ah),GPUsingle);
ex10f(A,B);
compareCPUGPU(Ah,A);

%%
IDX = GPUsingle([3 4 6 1]);
Ah = Bh(single(IDX));
A = zeros(size(Ah),GPUsingle);
ex10g(A,B,IDX);
compareCPUGPU(Ah,A);

%%
Ah = single(rand(100));
Bh = single(rand(1,10));
A = GPUsingle(Ah);
B = GPUsingle(Bh);
Ah(1:10) = Bh;
ex10h(A,B);
compareCPUGPU(Ah,A);

%%
Ah = Bh(1:10);
A = ex10i(B);
compareCPUGPU(Ah,A);



