% Save current directory
d = pwd;

% Compile GPUmat if it does not exist
if ~exist([d '/GPUmat'],'dir')
    cd([d '/gpumat-code']);
    addpath([d '/gpumat-code/util']);
    compile;
end

% Compile ShearCuda
cd(d);
setpaths;
GPUstart;
cd([d '/shearcuda']);
make cuda;
make cpp;
makemex;

% Compile 3D Shearlet
cd([d '/shearcuda3d/3DShearTrans']);
make cpp;

% Compile crack detection
cd([d '/crack_detection']);
make cuda
make cpp

% Done
cd(d);
