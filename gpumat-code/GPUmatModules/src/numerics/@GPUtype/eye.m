% eye - Identity matrix
% 
% SYNTAX
% 
% eye(N,CLASSNAME)
% eye(M,N,CLASSNAME)
% eye([M,N],CLASSNAME)
% eye(M,N,P,...?,CLASSNAME)
% eye([M N P ...],CLASSNAME)
% 
% CLASSNAME = GPUsingle/GPUdouble
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% EYE(M,N,CLASSNAME) or EYE([M,N],CLASSNAME) is an M-by-N ma-
% trix with 1's of class CLASSNAME on the diagonal and zeros else-
% where. CLASSNAME can be GPUsingle or GPUdouble
% Compilation supported
% 
% EXAMPLE
% 
% X = eye(2,3,GPUsingle);
% X = eye([4 5], GPUdouble);
