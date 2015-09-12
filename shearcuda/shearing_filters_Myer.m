function [shear_f]=shearing_filters_Myer(m,num,L,dataType)
% This function computes the directional/shearing filters using the Meyer window
% function.
% 
% Inputs: n1 - indicates the supports size of the directional filter is n1xn1
%         level - indicates that the number of directions is 2^level 
%         L - size of the input image ; L by L input image.
%         dataType - string containing the class of data
%             (single/double/GPUsingle/GPUdouble)
%         
% Output: if dataType is 'single' or 'double':
%             a sequence of 2D directional/shearing filters w_s where the
%             third index determines which directional filter is used
%         if dataType is 'GPUsingle' or 'GPUdouble':
%             a struct containing a sequence of 2D directional/shearing and
%             preallocated buffers for GPU processing. To avoid memory leaks,
%             it is necessary to call clear_shearing_filters_cuda on this
%             data structure before this object goes out of scope.
%
% Comments provided by Glenn R. Easley and Demetrio Labate. 
% Originally written by Glenn R. Easley on Feb 2, 2006.
% Copyright 2011 by Glenn R. Easley. All Rights Reserved.

% Modified by Xavier Gibert-Serra, May 2013
% Copyright 2013 University of Maryland. All Rights Reserved.

% generate indexing coordinate for Pseudo-Polar Grid

for j = 1:length(num)
    n1 = m(j); level = num(j);    
    [x11,y11,x12,y12,F1]=gen_x_y_cordinates(n1);

    wf=windowing(ones(2*n1,1),2^level);
    w_s{j}=zeros(n1,n1,2^level); %initialize window array
    for k=1:2^level,
        temp=wf(:,k)*ones(n1,1)';
        w_s{j}(:,:,k)=rec_from_pol(temp,n1,x11,y11,x12,y12,F1); % convert window array into Cartesian coord.
        w_s{j}(:,:,k)=real(fftshift(ifft2(fftshift(w_s{j}(:,:,k)))))./sqrt(n1); 
    end
end

isCPU = true;
if exist('dataType','var')
    isCPU = strcmp(dataType,'single') || strcmp(dataType,'double');

    % Convert w_s to desired data type
    switch(dataType)
        case 'single'
            for j=1:length(num)
                w_s{j} = single(w_s{j});
            end
        case 'GPUsingle'
            for j=1:length(num)
                w_s{j} = GPUsingle(w_s{j});
            end
        case 'GPUdouble'
            for j=1:length(num)
                w_s{j} = GPUdouble(w_s{j});
            end
    end
else
    dataType = 'double';
end

if isCPU
    shear_f=cell(1,length(num)); % declare cell array containing shearing filters
    for j=1:length(num),
        for k=1:2^num(j),
           shear_f{j}(:,:,k) =(fft2(w_s{j}(:,:,k),L,L)./L);
        end
    end
    for j=1:length(num), 
        d=sqrt(sum((shear_f{j}).*conj(shear_f{j}),3));
        for k=1:2^num(j),
            shear_f{j}(:,:,k)=shear_f{j}(:,:,k)./d;
        end
    end
else
    % In GPU mode, perform the rest of the calculations on the GPU
    shear_f = prepare_shearing_filters_cuda(w_s,L);
end
