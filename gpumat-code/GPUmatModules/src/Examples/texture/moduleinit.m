function moduleinit
% moduleinit Loads the user defined module. 

%% check GPUmat version
disp('- Loading module EXAMPLES_TEXTURE');
ver = 0.280;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE EXAMPLES_TEXTURE requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

[status,major,minor] = cudaGetDeviceMajorMinor(GPUgetActiveDeviceNumber);
cubin = ['texture' num2str(major) num2str(minor) '.cubin'];
disp(['  -> ' cubin ]);
GPUuserModuleLoad('examples_texture',['.' filesep cubin])

end
  
