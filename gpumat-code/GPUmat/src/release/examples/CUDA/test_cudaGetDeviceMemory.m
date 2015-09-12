function test_cudaGetDeviceMemory
dev = 0;
[status,totmem] = cudaGetDeviceMemory(dev);
if (status ~= 0)
  error(['Unable to get the total memory for device N. ' num2str(dev) ]);
end
disp(['    Total memory is ' num2str(totmem/(1024*1024)) 'MB']);

end