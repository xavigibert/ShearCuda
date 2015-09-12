%% Test GPUtype examples

%% Display properties of a GPUtype

% single/real
A = GPUsingle(rand(2,2,2,2));
gputype_properties(A);

% single/complex
A = complex(A);
gputype_properties(A);

% double/real
if (GPUisDoublePrecision)
  A = GPUdouble(rand(2,2,2,2));
  gputype_properties(A);
  
  % double/complex
  A = complex(A);
  gputype_properties(A);
end

%% Create GPUtype

% single/real
R = gputype_create1(0);

% single/complex
R = gputype_create1(1);

if (GPUisDoublePrecision)
  % double/real
  R = gputype_create1(2);
  
  % double/complex
  R = gputype_create1(3);
end

%% Create a GPUtype from a Matlab array
if (GPUisDoublePrecision)
  Ah = rand(100);
  A = gputype_create2(Ah);
end

Ah = single(rand(100));
A = gputype_create2(Ah);

%% Clones a GPUtype
A = GPUsingle(rand(10));
B = gputype_clone(A);

% getPtr prints the pointer to GPU memory
% A and B have different pointers
getPtr(A)
getPtr(B)


