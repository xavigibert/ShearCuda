function test_cudaGetDeviceCount
count = 0;
[status,count] = cudaGetDeviceCount(count);
if (status ~=0)
  error('Unable to get the number of devices');
end

count
end
