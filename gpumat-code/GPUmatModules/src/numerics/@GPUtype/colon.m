% colon - Colon
% 
% SYNTAX
% 
% R = colon(J,K,GPUsingle)
% R = colon(J,D,K,GPUsingle)
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% COLON(J,K,GPUsingle)        is   the    same     as   J:K    and
% COLON(J,D,K,GPUsingle) is the same as J:D:K. J:K is the
% same as [J, J+1, ..., K]. J:K is empty if J > K. J:D:K is the
% same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D). J:D:K
% is empty if D == 0, if D > 0 and J > K, or if D < 0 and J < K.
% 
% Compilation supported
% 
% EXAMPLE
% 
% A = colon(1,2,10,GPUsingle)
