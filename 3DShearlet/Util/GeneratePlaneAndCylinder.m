function  [ X ] = GeneratePlaneAndCylinder(radius, sideLength)

     X=ones(sideLength,sideLength,sideLength,'uint8')*255;
     for k=sideLength/2:sideLength
      for j=1:sideLength
        for i=1:sideLength
            if ((i-sideLength/2)^2+(j-sideLength/2)^2 <= radius^2)
                 X(i,j,k)=1;
            end
        end
      end
     end
     
     X(:,:,sideLength/2:sideLength/2+3)=1;

end