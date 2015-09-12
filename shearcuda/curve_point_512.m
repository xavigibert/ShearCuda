function [img nimg] = curve_point(N,sigma)


% generating artificial test image with points and curves
% N : the number of points;
% sigma : gaussian noise level 

% Written by Wang-Q Lim on May 5, 2010. 
% Copyright 2010 by Wang-Q Lim. All Right Reserved.

% generates random points 
x = rand(N,2);
x = floor(500*x);

img = zeros(513,513);

% generates a circle
for row = 1:512
    for  col = 1:512
        img(row,col) = 255/(1+sqrt(((row-1)/511)^2+((col-1)/511)^2))^3;
    end
end


for i = 1:512
    for j = 1:512
        r = sqrt((i/512-1/2)^2+(j/512-1/2)^2); 
        if r > 0.35 && r < 0.355
            img(i,j) = 255;
        end
    end
end
            


% generates lines 
for j = 1:512
    
    img(min(round(1/2*j+10),513),j) = 255;
    img(min(round(1/2*j+11),513),j) = 255;
    
    img(min(round((513-j)+10),513),j) = 255;
    img(min(round((513-j)+11),513),j) = 255;
    
    
    img(min(round(2*j+264),513),j) = 255;
    img(min(round(2*j+265),513),j) = 255;
    
    
    img(j,min(round(1/2*j+300),513)) = 255;
    img(j, min(round(1/2*j+301),513)) = 255;
    
    img(j,min(round(1/3*j+160),513)) = 255;
    img(j,min(round(1/3*j+161),513)) = 255;
    
    img(j,min(round(1/3*(513-j)+100),513)) = 255;
    img(j, min(round(1/3*(513-j)+101),513)) = 255;
    
    img(j,160) = 255;
    img(j,159) = 255;
    
    img(240,j) = 255;
    img(241,j) = 255;
     
end

            


for j = 1:N
    img(max([x(j,1),1]),max([x(j,2),1])) = 255;
    img(max([x(j,1),1]),max([1,x(j,2)])) = 255;
    img(max([x(j,1)+1,1]),max([x(j,2),1])) = 255;
    img(max([x(j,1),1]),max([x(j,2)-1,1])) = 255;
    img(max([x(j,1),1]),max([x(j,2)+1,1])) = 255;
end

img = img(1:512,1:512);
img = min(img,255);

img = re_weight_band(img,[1 0]);
B=randn(512,512);
B=B*sigma;
nimg = img+B;