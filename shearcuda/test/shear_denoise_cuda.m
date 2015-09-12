function y = shear_denoise_cuda(nimg,sigma,shear,E,sc)
% Image denoising with Fourier based shearlet transform + Hard thresholding. 
%
% Input : nimg : noisy image.  
%         sigma : std of Gaussian Noise. 
%         shear : cell array of shearlet filters computed 
%                 from shearing_filters_Myer.m
%                 (see shearing_filters_Myer.m)          
%         E : l^2 norm of shearlets (see com_norm.m).
%         sc : a row vector of threshold parameters.
%              (see thresh.m)


GPUsync;
tic;
d=shear_trans_cuda(nimg,'maxflat',shear);
% apply forward shearlet transform.
thresh(d,sigma,2,E,sc,0);
% apply hard threshold on the shearlet coefficients.
y = inverse_shear_cuda(d,'maxflat',shear);
% apply inverse shearlet transform.
GPUsync;
toc;
