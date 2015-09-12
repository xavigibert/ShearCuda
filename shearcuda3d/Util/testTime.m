tic;
Theta =zeros(96,96,96,'single');
Phi =zeros(96,96,96,'single');
[T P]=GenerateSphericalFilterAngle(9);
shCoeff=ones(96,96,96,18,18);
shData=zeros(18,18);
xS=int8(96);
yS=int8(96);
zS=int8(96);
l1Sz=int8(18);
l2Sz=int8(18);
for x=1:xS
  for y=1:yS
    for z=1:zS   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
%For each point determine point of max orientation      
%       for l2=1:l2Sz
%         for l1=1:l1Sz
%          if l1 >=l1Sz/2 && l2>=l2Sz/2
%            shData(l2,l1)=0;
%           continue;
%          end
%          %plot3(l1,l2,abs(shCoeff{l2,l1}(x,y,z)),'r+:')
%          %surf(l1,l2,abs(shCoeff{l2,l1}(x,y,z)),'*');
%          shData(l2,l1)=abs(shCoeff{l2,l1}(y,x,z));
%        end
%       end

      
     % [V,I]= max(shData(:));
     [~ , I]= max(abs(shCoeff(y,x,z,:)));
                
     [l2 , l1] =ind2sub([l2Sz l1Sz],I);
     %[l2,l1] = ind2sub(size(shData),I); %To dochnage to fix vector in size
     Theta(y,x,z)=T(l2,l1);
     Phi(y,x,z)=P(l2,l1);
      % end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
  
    end
  end
end
toc
