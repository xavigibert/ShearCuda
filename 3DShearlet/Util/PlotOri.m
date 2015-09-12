function PlotOri(x,y,z,shCoeff)
[l2Sz  l1Sz]=size(shCoeff);

% l1=1:l1Sz;
% l2=1:l2Sz;
% mesh(abs(shCoeff{l2,l1}(x,y,z)));
surfData=zeros(l2Sz,l1Sz);
%hold
for l2=1:l2Sz
  for l1=1:l1Sz
    if l1 >=l1Sz/2 && l2>=l2Sz/2
      continue;
    end
      %plot3(l1,l2,abs(shCoeff{l2,l1}(x,y,z)),'r+:')
      %surf(l1,l2,abs(shCoeff{l2,l1}(x,y,z)),'*');
      surfData(l2,l1)=abs(shCoeff{l2,l1}(x,y,z));
  end
end
surf(surfData);
[~,I]= max(surfData(:));
[l2,l1]=ind2sub(size(surfData),I)
surfData(l2,l1)
surfData(I)=0;
[~,I]= max(surfData(:));
[l2,l1]=ind2sub(size(surfData),I)
surfData(l2,l1)
