function test_cudaGetDeviceMultProcCount
dev = 0;
[status,count] = cudaGetDeviceMultProcCount(dev);
if (status ~=0)
  error(['Unable to get the numer of multi proc. for device N. ' num2str(dev) ]);
end
disp(['    Mult. processors = ' num2str(count) ]);

end