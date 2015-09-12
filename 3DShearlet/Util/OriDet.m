% Generate Angle information f
%For the points in the image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function F= GenerateSphericalFilterAngle(numDir)
% Generates windowing filters
%Input:   
%        numDir     : shearlet Coefficient information
%Output: 
%        Theta ,Phi : Spherical Coordinate Angle
%                   : 
%Author             : P. S. Negi 5 March 2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Theta Phi]= OriDet(shCoeff)

%for each point get the index of maximum response direction(indexes)
%ignore the point below certain threshold

[yS,xS,zS]=size(shCoeff{1,1});
[l2Sz  l1Sz]=size(shCoeff);

% thr=(max(abs(shCoeff{floor(l2Sz/4),floor(l1Sz/4)}(:)))+max(abs(shCoeff{floor(l2Sz/4),floor((l1Sz/4)*3)}(:)))+...
%     max(abs(shCoeff{floor((l2Sz/4)*3),floor(l1Sz/4)}(:))))/(3*);
thr=10^(-4);

Theta =zeros(xS,yS,zS,'single');
Phi =zeros(xS,yS,zS,'single');
[T P]=GenerateSphericalFilterAngle(l1Sz/2);

shData=zeros(l2Sz,l1Sz);
for x=1:xS
  for y=1:yS
    for z=1:zS   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
%For each point determine point of max orientation      
      for l2=1:l2Sz
        for l1=1:l1Sz
         if l1 >=l1Sz/2 && l2>=l2Sz/2
           shData(l2,l1)=0;
          continue;
         end
         %plot3(l1,l2,abs(shCoeff{l2,l1}(x,y,z)),'r+:')
         %surf(l1,l2,abs(shCoeff{l2,l1}(x,y,z)),'*');
         shData(l2,l1)=abs(shCoeff{l2,l1}(y,x,z));
       end
      end
      
      [V,I]= max(shData(:));
      %if V>thr
        [l2,l1] = ind2sub([l2Sz l1Sz ],I); %To dochnage to fix vector in size
        Theta(y,x,z)=T(l2,l1);
        Phi(y,x,z)=P(l2,l1);
      % end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
  
    end
  end
end
