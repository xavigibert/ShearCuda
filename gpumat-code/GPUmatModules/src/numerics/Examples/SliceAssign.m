function SliceAssign
%% Examples from Wiki page
%% http://sourceforge.net/apps/mediawiki/gpumatmodules/index.php?title=Indexed_references

% Although the elements of a GPU variable can be accessed as any other
% Matlab array, the functions slice and assign are faster. They have 
% a syntax similar to the standard Matlab indexing. Some examples below. 

%% Slice examples

Bh = single(rand(100));
B = GPUsingle(Bh);

%%
Ah = Bh(1:end);
A = slice(B,[1,1,END]);
compareCPUGPU(Ah,A);

%%
Ah = Bh(1:10,:);
A = slice(B,[1,1,10],':');
compareCPUGPU(Ah,A);

%%
Ah = Bh([2 3 1],:);
A = slice(B,{[2 3 1]},':');
compareCPUGPU(Ah,A);

%%
Ah = Bh([2 3 1],1);
A = slice(B,{[2 3 1]},1);
compareCPUGPU(Ah,A);

%%
Ah = Bh(:,:);
A = slice(B,':',':');
compareCPUGPU(Ah,A);

%% Assign Examples
A = GPUsingle(rand(100));
B = GPUsingle(rand(10,10));
Ah = single(A);
Bh = single(B);

%A(1:10,1:10) = B;
assign(1, A, B, [1,1,10],[1,1,10]);
Ah(1:10,1:10) = Bh;
compareCPUGPU(Ah,A);

%%
A = GPUsingle(rand(100));
B = GPUsingle(rand(4,10));
Ah = single(A);
Bh = single(B);

%A([2 3 1 5],1:10) = B;
assign(1, A, B, {[2 3 1 5]},[1,1,10]);
Ah([2 3 1 5],1:10) = Bh;
compareCPUGPU(Ah,A);


