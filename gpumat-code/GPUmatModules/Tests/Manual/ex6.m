function ex6
A = GPUsingle();       %empty constructor
setSize(A,[100 100]);  %set variable size
setReal(A);            %set variable as real
GPUallocVector(A);     %allocate on GPU memory

% above commands are similar to
A = zeros([100 100], GPUsingle);

% If we need a complex variable:
A = complex(zeros([100 100], GPUsingle));

