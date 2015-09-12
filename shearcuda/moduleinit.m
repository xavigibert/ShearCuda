function moduleinit
% moduleinit Loads the user defined module. 

%% check GPUmat version
disp('- Loading module SHEARCUDA');
ver = 0.280;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE SHEARCUDA requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(GPUgetActiveDeviceNumber);
cubin = ['shear_cuda' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('shear_cuda',['.' filesep cubin])

end
  
