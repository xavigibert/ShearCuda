function ex5
A = GPUsingle();       %empty constructor
setSize(A,[100 100]);  %set variable size
setReal(A);            %set variable as real
GPUallocVector(A);     %allocate on GPU memory

A = GPUsingle();        %empty constructor
setSize(A,[10 10]);     %set variable size
setComplex(A);          %set variable as complex
GPUallocVector(A);      %allocate on GPU memory

