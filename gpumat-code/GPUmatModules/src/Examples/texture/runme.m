function runme
%% run test
moduleinit

% double
if (GPUisDoublePrecision)
  A = GPUdouble([0 2 4 6]);
  I = GPUdouble([0 2 1 3]);
  texture_lininterp(A,I)
end


% single
A = GPUdouble([0 2 4 6]);
I = GPUdouble([0 2 1 3]);
texture_lininterp(A,I)


end