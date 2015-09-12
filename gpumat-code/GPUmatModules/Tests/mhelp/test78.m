function test78
%% Create a GPUtype from a Matlab array
if (GPUisDoublePrecision)
  Ah = rand(100);
  A = gputype_create2(Ah);
end

Ah = single(rand(100));
A = gputype_create2(Ah);
