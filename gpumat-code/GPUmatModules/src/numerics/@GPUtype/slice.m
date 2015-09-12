% slice - Subscripted reference
% 
% SYNTAX
% 
% R =   slice(X, R1, R2, ..., RN)
% X -   GPUsingle, GPUdouble
% R1,   R2, ..., RN - Range
% R -   GPUsingle, GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% slice(X, R1,...,RN) is an array formed from the elements of X
% specified by the ranges R1, R2, RN. A range can be constructed as
% follows:
% [inf,stride,sup] - defines a range between inf and sup with spec-
% ified stride. It is similar to the Matlab syntax A(inf:stride:sup). The
% special keyword END (please note, uppercase END) can be used.
% ':' - similar to the colon used in Matlab indexing.
% {[i1, i2, ..., in]} -any array enclosed by brackets is consid-
% ered an indexes array, similar to A([1 2 3 4 1 2]) in Matlab.
% i1 - a single value is interpreted as an index. Similar to A(10) in
% Matlab.
% 
% Compilation supported
% 
% EXAMPLE
% 
% Bh = single(rand(100));
% B = GPUsingle(Bh);
% Ah = Bh(1:end);
% A = slice(B,[1,1,END]);
% Ah = Bh(1:10,:);
% A = slice(B,[1,1,10],':');
% Ah = Bh([2 3 1],:);
% A = slice(B,{[2 3 1]},':');
% Ah = Bh([2 3 1],1);
% A = slice(B,{[2 3 1]},1);
% Ah = Bh(:,:);
% A = slice(B,':',':');
