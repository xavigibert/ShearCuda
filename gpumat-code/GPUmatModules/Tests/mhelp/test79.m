function test79
% single/real
A = rand(2,2,2,2,GPUsingle);
gputype_properties(A);

% single/complex
A = complex(A);
gputype_properties(A);

% double/real
if (GPUisDoublePrecision)
  A = rand(2,2,2,2,GPUdouble);
  gputype_properties(A);
  
  % double/complex
  A = complex(A);
  gputype_properties(A);
end
