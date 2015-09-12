% Estimation of the standard deviation of an image assuming that the noise
% is Gaussian white additive. The file estimates the standard deviation
% using the median filter on the fine scale subband of the wavelet
% decomposition.
 function sd_estimate=sdest3D(x)
 sd_estimate=0;
 totalFrames=size(x,3);
 for i=1:totalFrames
   [ca,ch,cv,cd] = dwt2(x(:,:,i),'sym4','mode','sym');
   sd_estimate = sd_estimate+ mad(cd(:),1)/(totalFrames* 0.6745);
   
 end

