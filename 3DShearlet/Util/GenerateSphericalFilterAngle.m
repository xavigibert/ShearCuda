% Generate Angle information for the filter locations
%Take odd number of filter so that horizontal or vertical orientation can
%be accuratley captured
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function F= GenerateSphericalFilterAngle(numDir)
% Generates windowing filters
%Input:   
%        numDir     : number of direction in L1 and L2
%Output: 
%        Theta ,Phi : Spherical Coordinate Angle
%                   : 
%Author             : P. S. Negi 5 March 2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Theta , Phi]= GenerateSphericalFilterAngle(numDir)

Theta=zeros(2*numDir,2*numDir);
Phi=zeros(2*numDir,2*numDir);
gridLength=2*numDir +1;
%For horizontal pyramidal cone with X as major Axis
x=floor(gridLength/2);
for y=-(floor(gridLength/2)-1):2:(floor(gridLength/2)-1) %controls l1
  l1= ((y+numDir) -1)/2+1;
  for z=(floor(gridLength/2)-1):-2:-(floor(gridLength/2)-1) % controla l2    
    l2= ((-z+numDir) -1)/2+1;    
    Theta(l2,l1)=atan2(y,x);
    Phi(l2,l1)=acos(z/sqrt(x^2+y^2+z^2));    
  end
end
%for pyramidal cone with majoe axis Y
y=floor(gridLength/2);
for x=(floor(gridLength/2)-1):-2:-(floor(gridLength/2)-1) %controls l1
  l1= ((-x+numDir) -1)/2+1;
  for z=floor(gridLength/2)-1:-2:-(floor(gridLength/2)-1) % controls l2    
    l2= ((-z+numDir) -1)/2+1;    
    Theta(l2,numDir+l1)=atan2(y,x);
    Phi(l2,l1+numDir)=acos(z/sqrt(x^2+y^2+z^2));    
  end
end
%for pyramidal Cone with Major Axia Z
z=floor(gridLength/2);
for y=(floor(gridLength/2)-1):-2:-(floor(gridLength/2)-1) %controls l1
  l1= ((-y+numDir) -1)/2+1;
  for x=-(floor(gridLength/2)-1):2:(floor(gridLength/2)-1) % controls l2    
    l2= ((x+numDir) -1)/2+1;    
    Theta(l2+numDir,l1)=atan2(y,x);
    Phi(l2+numDir,l1)=acos(z/sqrt(x^2+y^2+z^2));    
  end
end
%convert anything in lower sphere o upper sphere by
%Theta ->Theat +pi %2*pi and Phi->pi - Phi

[LI2, LI1 , ~]=find(Phi >pi/2);
Phi(LI2, LI1) = pi - Phi(LI2, LI1);
Theta(LI2,LI1)= mod(Theta(LI2,LI1)+pi,2*pi);
[LI2, LI1, ~]=find(Theta<0);
Theta(LI2,LI1)= mod(Theta(LI2,LI1),2*pi);
%convert Theta from -pi<=Theta<pi  to 0<=Theta<2*pi
% neg=Theta <0;
% Theta=(2*pi+Theta).*neg+ Theta.*~neg;
Theta=Theta*180/pi;
Phi=Phi*180/pi;








