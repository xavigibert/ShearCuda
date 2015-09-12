function []=init(gpu_device)

if ~exist('gpu_device','var')
    gpu_device = -1;
end
setpaths
myGPUstart(gpu_device);
cd shearcuda
moduleinit
cd ../crack_detection
moduleinit

cd ..