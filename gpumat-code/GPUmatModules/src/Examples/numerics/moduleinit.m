function moduleinit
% moduleinit Loads the user defined module. 

%% check GPUmat version
disp('- Loading module EXAMPLES_NUMERICS');
ver = 0.280;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE EXAMPLES_NUMERICS requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(GPUgetActiveDeviceNumber);
cubin = ['numerics' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('examples_numerics',['.' filesep cubin])

end
  
