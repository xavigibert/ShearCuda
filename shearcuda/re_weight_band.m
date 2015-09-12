function [image1 y] = re_weight_band(P,coeff)

% This routine generates image reconstructed with different weights for 
% band passed images. 
% 
% Output : reconstructed image with weighting across frequency bands.
%
% Input : P : input image 
%         coeff : weights across frequency bands ---> coeff = [coeff(1),...,coeff(L)]
%         coeff(1) weight for low frequency... coeff(L) wight for high frequency. 
%
% Let B(1)...B(L) : band pass filters for frequency decomposition
%     B^*(1),...,B^*(L) : band pass filters for reconstruction.
% Especially, we have B^*(1)*B(1)*I + ... + B^*(L)*B(L)*I = I for given 
% image I where * is 2D convolution. 
% Then output image P is simply obtained by
% P = coeff(1)B^*(1)*B(1)*I + ... + coeff(L)B^*(L)*B(L)*I. 

% Written by Wang-Q Lim on May 5, 2010. 
% Copyright 2010 by Wang-Q Lim. All Right Reserved.


y = atrousdec(P,'maxflat',length(coeff)-1);

for  j = 1:length(coeff)
    y{j} = coeff(j)*y{j};
end

image1 = atrousrec(y,'maxflat');
    
