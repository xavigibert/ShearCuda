function moduleinit
% moduleinit Loads the user defined module. 

%% check GPUmat version
disp('- Loading module CRACKSCUDA');
ver = 0.280;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE CRACKSCUDA requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(GPUgetActiveDeviceNumber);
cubin = ['cracks_cuda' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('cracks_cuda',['.' filesep cubin])

end
  
