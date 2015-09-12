function moduleinit

%% check GPUmat version
disp('- Loading module NUMERICS');
ver = 0.280;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE NUMERICS requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end
[status,major,minor] = cudaGetDeviceMajorMinor(GPUgetActiveDeviceNumber);
cubin = ['numerics' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('numerics',['.' filesep cubin])

% Load NUMERICS module manager
NumericsModuleManager;
end
  
