A = zeros(5,GPUsingle);
GPUfill(A, 1, 0, 0, 0, 0, 0);

A = zeros(5,GPUsingle);
GPUfill(A, 1, 0, 0, 2, 0, 0);

A = zeros(5,GPUsingle);
GPUfill(A, 1, 0, 0, 2, 1, 0);

A = zeros(5,GPUsingle);
GPUfill(A, 1, 1, numel(A), 0, 0, 0);

A = zeros(5,GPUsingle);
GPUfill(A, 1, 1, numel(A), 2, 0, 0);

A = zeros(2,complex(GPUsingle));
GPUfill(A, 1, 1, numel(A), 0, 0, 2);

A = zeros(2,complex(GPUsingle));
GPUfill(A, 1, 1, numel(A), 0, 0, 1);
