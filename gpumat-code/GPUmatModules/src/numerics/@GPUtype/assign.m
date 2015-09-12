% assign - Indexed assignement
% 
% SYNTAX
% 
% assign(dir, P, Q. R1, R2, ..., RN)
% P - GPUsingle, GPUdouble
% Q - GPUsingle, GPUdouble, Matlab (scalar supported)
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% ASSIGN(DIR, P, Q, R1, R2, ..., RN) performs the following
% operations, depending on the value of the parameter DIR:
% DIR = 0 -> P = Q(R1, R2, ..., RN)
% DIR = 1 -> P(R1, R2, ..., RN) = Q
% R1, R2, RN represents a sequence of ranges. A range can be con-
% structed as follows:
% [inf,stride,sup] - defines a range between inf and sup with spec-
% ified stride. It is similar to the Matlab syntax A(inf:stride:sup). The
% special keyword END (please note, uppercase END) can be used.
% ':' - similar to the colon used in Matlab indexing.
% {[i1, i2, ..., in]} -any array enclosed by brackets is consid-
% ered an indexes array, similar to A([1 2 3 4 1 2]) in Matlab.
% i1 - a single value is interpreted as an index. Similar to A(10) in
% Matlab.
% Compilation supported
% 
% EXAMPLE
% 
% A = rand(100,GPUsingle);
% B = rand(10,10,GPUsingle);
% Ah = single(A);
% Bh = single(B);
% Ah(1:10,1:10) = Bh;
% assign(1, A, B, [1,1,10],[1,1,10]);
% assign(1, A, Bh, [1,1,10],[1,1,10]);
% assign(1, A, single(10), [1,1,10],[1,1,10]);
