function GPUfill

%% Fill with ones
A = zeros(5,GPUsingle);
GPUfill(A, 1, 0, 0, 0, 0, 0);
A

%% Fill with ones, and element every 2
A = zeros(5,GPUsingle);
GPUfill(A, 1, 0, 0, 2, 0, 0);
A

%% Fill with ones, and element every 2
%  starting from the 2nd element
A = zeros(5,GPUsingle);
GPUfill(A, 1, 0, 0, 2, 1, 0);
A

%% Fill with a sequence of numbers from 1 to numel(A)
A = zeros(5,GPUsingle);
GPUfill(A, 1, 1, numel(A), 0, 0, 0);
A

%% Fill with a sequence of numbers from 1 to numel(A)
%  An element every 2 is modified
A = zeros(5,GPUsingle);
GPUfill(A, 1, 1, numel(A), 2, 0, 0);
A

%% Use parameter type=2 to modify both real and complex part
A = zeros(2,complex(GPUsingle));
GPUfill(A, 1, 1, numel(A), 0, 0, 2);
A

%% Use parameter type=1 to modify only the complex part
A = zeros(2,complex(GPUsingle));
GPUfill(A, 1, 1, numel(A), 0, 0, 1);
A

