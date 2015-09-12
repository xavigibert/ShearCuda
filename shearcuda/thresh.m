function [coeff maxi] = thresh(Ct,lambda,option,E,sc,opt)

% This routine performs Hard Thresholding for shearlets (or wavelets)

% Input :
%Ct : transform coefficients (wavelets or shearlets)
%lambda : threshold parameter (any positive number)
%option = 1 : thresholding for wavelets, option = 2 : thesholding for shearlets
%E: l^2 norm of shearlets (see com_norm.m)
%
%sc: row vector [s(0),s(1),...,s(L)] where each entry s(j) is a
%    thresholding parameter for each scale j. 
%Here, j = 0 is a scale of low frequency and  j = L is the finest scale.   
%
%This routine applies Hard Threshoding on Each shearlet coefficient Ct(j,k,n)
%with thresholding parameter sc(j)*lambda across scales j.  
%
%opt : opt == 1 : compute max magnitude of coeff otherwise return 0 for output maxi.

% Output :
%coeff : coefficients with (Hard) thresholding.
%maxi : maximum magnitude of coefficients. 


% Written by Wang-Q Lim on May 5, 2010. 
% Copyright 2010 by Wang-Q Lim. All Right Reserved.

temp = [];
if option == 2    
    % Apply thresholding
    for s = 1:length(Ct)
          for w = 1:size(Ct{s},3)
              Ct{s}(:,:,w) = Ct{s}(:,:,w).* (abs(Ct{s}(:,:,w)) > sc(s)*(lambda)*E(s,w));
              % apply hard thresholding on each shearlet coefficient with threshold 
              % parameter sc(s)*lambda for each scale s. 
              if opt == 1
                  wedge = max(max(abs(Ct{s}(:,:,w)/E(s,w))));
                  temp = [temp; wedge(:)];
              end
          end
    end
else

    Ct{2} = Ct{2}.*(abs(Ct{2})>lambda);
    % apply hard thresholding on wavelet coeffcients with threshold parameter lambda. 
    if opt == 1
        temp = abs(Ct{2}(:));
    end
end
coeff = Ct;
if opt == 1
    maxi = max(temp);
else
    maxi = 0;
end
    
    
                
