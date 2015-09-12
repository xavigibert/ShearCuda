function moduleinit
% moduleinit Loads the user defined module. 

%% check GPUmat version
disp('- Loading module RAND');
ver = 0.280;
gver = GPUmatVersion;
if (str2num(gver.version)<ver)
  warning(['MODULE RAND requires GPUmat version ' num2str(ver) ' or higher. UNABLE TO LOAD MODULE']);
  return;
end

%% Module manager
RANDModuleManager
end
  
