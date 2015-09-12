function test77
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
