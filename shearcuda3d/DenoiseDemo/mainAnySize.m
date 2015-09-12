%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sample routine showing the use of shearlet 3d functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;
%matlabpool open 3 
addpath('../CONVNFFT_Folder');
addpath('../3DBP');
addpath('../3DShearTrans');
addpath('../Util');
addpath('../Data');
dataClass='single';% 'single' or 'double'

%Thresholding multiplier for hardthresholding
%T=ones(level+1,1)*3.08;
T=[ 3.3 3.0 3.0 3.0];

%Noise Simulated

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sigma=50;
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
dBand={{[8 8]}, ... %%%%for level =1
        {[8 8],[4 4]}, ...  %%%% for level =2
        {[9 9], [5 5],[5 5]}, ...   %%%% for level =3
        {[8 8],[8 8],[4 4],[4 4]}}; %%%%% for level =4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filterSize=[28 31 31];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load Data and introduce some noise for simulation

%load tempete
%load mobile2_
load mobile2_sequence95X95X45
% load movingCell

x= double(MarshalData(X,3));

% sigma=sdest3D(x);
sigma=20;
% x=double(X);
% xn = x + sigma * randn(size(x));
xn=x;
fprintf('introduced  PSNR %f\n',PSNR(x,xn));

%Build Windowing Filter for different Band
F= GetMeyerBasedFilter(level,dBand,filterSize,'double');

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

%Compute Shearlet Coefficient, Threshold and denoise.
%If large memory is available then can collect all the
%respective pyramidal cone data in a 1X3 cell and can do
%further processing in a single function


for pyrCone=1:3
  shCoeff=ShDec(pyrCone,F,BP,level,dataClass);
  %clear F{pyrCone,:}
  shCoeff=ThresholdShCoeff(shCoeff,nsstScalars,pyrCone,sigma,T);
  partialBP{pyrCone}=ShRec(shCoeff);
  %clear shCoeff;
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

recBP{level+1}=BP{level+1};
%Reconstruct
xRec=DoPyrRec(recBP);
fprintf('Reconstuction error after coeff suppression %e   PSNR %f\n', ... 
  froNormMatn(x,xRec),PSNR(x,xRec));
%matlabpool close
xRec=unMarshal(xRec,size(X));
toc


