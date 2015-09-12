%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sample routine showing the use of shearlet 3d functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;
%matlabpool open 3 
addpath('../CONVNFFT_Folder');
addpath('../3DBP');
addpath('../3DShearTrans');
addpath('../Util');
addpath('../Data');
dataClass='GPUsingle';% 'single', 'double', 'GPUsingle', or 'GPUdouble'
isGPU=strcmp(dataClass,'GPUsingle') || strcmp(dataClass,'GPUdouble');
doGpuTiming = true;

%Thresholding multiplier for hardthresholding
%T=ones(level+1,1)*3.08;
T=[ 3.3 3.0 3.0 3.0];

if isGPU
    if doGpuTiming
        shear_timers(3);
    else
        shear_timers(4);
    end        
end

%Noise Simulated

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma=30;
level=3; % choose level of decomposition ,
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%As per the level choosen different number of directinal band wil be used.
%if level =1, then there will be 8X8 band in each of 3 pyramidal zone for finest level 1
%if level=2, finest level 1 will have 8X8 band and next coarser level 2 
%will have 4X4 in each of the 3 pyramidal zone 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%NOTE::THESE ARE JUST SUGGESTIVE DIRECTIONAL BAND COMPOSITION
%% USER MAY TRY TO PLAY WITH THEM 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Cell to specify different directional band at as per level choosen
%In current implementation specifying second number of direction is
%ignored 
dBand={{[ 9 9]}, ... %%%%for level =1
        {[2 2 ],[2 2]}, ...  %%%% for level =2
        {[8 8 ], [4 4],[4 4]}, ...   %%%% for level =3
        {[8 8],[8 8],[4 4],[4 4]}}; %%%%% for level =4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load Data and introduce some noise for simulation

%load tempete
% load mobile2_sequence
% load Cube48InsideCube96
%load mobile_sequence96X96X96
%load coastguard_sequence96x96x96
%load oilPainting
load waterfall
%choose filter size such that remainder is 1 when divided by number of
%direction in that level
filterSize=[28 31 31];
x=double(X);
xn = x + sigma * randn(size(x));
%if strcmp(dataClass,'single')
%    x = single(x);
%    xn = single(xn);
%end
fprintf('introduced  PSNR %f\n',PSNR(x,xn));

%Build Windowing Filter for different Band
if strcmp(dataClass, 'single')
    F= GetMeyerBasedFilter(level,dBand, filterSize ,'single');
    xn = single(xn);
else
    F= GetMeyerBasedFilter(level,dBand, filterSize ,'double');
end

%Do the Band Pass of noisy Data
BP=DoPyrDec(xn,level);

%for storing partial reconstructed bandpass data
partialBP=cell(size(BP));
recBP=cell(size(BP));

% Determines via Monte Carlo the standard deviation of
% the white Gaussian noise with for each scale and
% directional components when a white Gaussian noise of
% standard deviation of 1 is feed through.
nsstScalarFileName=['nsstScalarsData' regexprep( num2str( [level dBand{level}{:}]) ,'[^\w'']','') '.mat'];
if exist(nsstScalarFileName,'file')
load(nsstScalarFileName);
else 
nsstScalars = NsstScalars(size(x),F,dataClass);
save(nsstScalarFileName, 'nsstScalars');
end

tic

if isGPU
    % Transfer inputs to GPU
    F = deepCopyToGpu(F);
    BP = deepCopyToGpu(BP);
end

%Compute Shearlet Coefficient, Threshold and denoise.
%If large memory is available then can collect all the
%respective pyramidal cone data in a 1X3 cell and can do
%further processing in a single function

if ~isGPU
  for pyrCone=1:3
    shCoeff=ShDec(pyrCone,F,BP,level,dataClass);
    shCoeff=ThresholdShCoeff(shCoeff,nsstScalars,pyrCone,sigma,T);
    partialBP{pyrCone}=ShRec(shCoeff);
  end
  
  clear F;

  clear shCoeff;

  %%%%Sum Up different pyramidal cone Band Pass Data%%


  for l=1:level      
    %% Assuming different pyramidal zone have same shCoeff size at different 
    %%level
    recBP{l}=zeros(size(partialBP{1}{l}),dataClass);
    for pyrCone =1:3
      recBP{l}=recBP{l}+ partialBP{pyrCone}{l};
    end
  end
else
  % Due to limitations on GPU memory, we cannot hold all the shearlet
  % coefficients at once. We need to apply thresholding and reconstruct
  % each directional filter separately
  recBP=ShDecThRec_cuda(F,BP,level,dataClass,nsstScalars,sigma,T);
end
    
recBP{level+1}=BP{level+1};

if isGPU
    % Transfer outputs from GPU
    recBP = deepCopyFromGpu(recBP);
    GPUsync;
end
toc

%Reconstruct
xRec=DoPyrRec(recBP);
fprintf('Reconstruction error after coeff suppression %e   PSNR %f\n', ... 
  froNormMatn(x,xRec),PSNR(x,xRec));
%matlabpool close

if isGPU & doGpuTiming
    shear_timers(0);    % Display timers
    shear_timers(1);    % Reset timer
end

toc

