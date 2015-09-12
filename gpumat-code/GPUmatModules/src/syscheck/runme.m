function runme
% start CUDA driver
cudrv

% check CUBLAS
cublas

[status,devcount]=cudaGetDeviceCount(10);

dev = 0;
info
if (devcount>1)
  disp('  - Your system has multiple GPUs installed'); 
  dev = input(['    -> Please specify the GPU device number to use [0-' num2str(devcount-1) ']: ']); 
  if (dev>=devcount)
    error('Specified device number is incorrect');
  end  
end

% load kernel
loadkernel(dev)

