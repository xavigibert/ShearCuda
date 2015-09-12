function test_cudaGetDeviceMajorMinor
dev = 0;
[status,major,minor] = cudaGetDeviceMajorMinor(dev);
if (status ~= 0)
  error(['Unable to get the compute capability']);
end

major
minor

end