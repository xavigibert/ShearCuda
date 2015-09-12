% GPUfill - Fill a GPU variable
% 
% SYNTAX
% 
% GPUfill(A, offset, incr, m, p, offsetp, type)
% A - GPUsingle, GPUdouble
% offset, incr, m, p, offsetp, type - Matlab
% 
% 
% MODULE NAME
% NUMERICS
% 
% DESCRIPTION
% GPUfill(A, offset, incr, m, p, offsetp, type) fills an ex-
% isting array with specific values.
% Compilation supported
% 
% EXAMPLE
% 
% %% Fill with ones
% A = zeros(5,GPUsingle);
% GPUfill(A, 1, 0, 0, 0, 0, 0);
% %% Fill with ones, and element every 2
% A = zeros(5,GPUsingle);
% GPUfill(A, 1, 0, 0, 2, 0, 0);
% %% Fill with ones, and element every 2
% % starting from the 2nd element
% A = zeros(5,GPUsingle);
% GPUfill(A, 1, 0, 0, 2, 1, 0);
% %% Fill with a sequence of numbers from 1 to numel(A)
% A = zeros(5,GPUsingle);
% GPUfill(A, 1, 1, numel(A), 0, 0, 0);
% %% Fill with a sequence of numbers from 1 to numel(A)
% % An element every 2 is modified
% A = zeros(5,GPUsingle);
% GPUfill(A, 1, 1, numel(A), 2, 0, 0);
% %% type=2 to modify both real and complex part
% A = zeros(2,complex(GPUsingle));
% GPUfill(A, 1, 1, numel(A), 0, 0, 2);
% %% Modify only the complex part
% A = zeros(2,complex(GPUsingle));
% GPUfill(A, 1, 1, numel(A), 0, 0, 1);
