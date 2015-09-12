function save_shearing_filters(m,num,dataType)
% This function computes the shearing filters (wedge shaped) using the Meyer window
% function and saved them to a file so they can be used outside MATLAB
%
% Inputs: m - size of shearing filter matrix desired, m = [m(1),...,m(N)] where
%             each entry m(j) determines size of shearing filter matrix at scale j. 
%         num - the parameter determining the number of directions. 
%               num = [num(1),...,num(N)] where each entry num(j)
%               determines the number of directions at scale j.  
%               num(j) ---> 2^(num(j)) + 2 directions.
%         L - size of the input image ; L by L input image. 
%
%
% Outputs: dshear{j}(:,:,k) - m(j) by m(j) shearing  filter matrix at orientation
%                             k and scale j.  
%
% For example, save_shearing_filters([32 32 16 16],[3 3 4 4],'single');
% produces cell array 'dshear' consisting of 
%          10 shearing filters (32 by 32) at scale j = 1 (coarse scale)  
%          10 shearing filters (32 by 32) at scale j = 2 
%          18 shearing filters (16 by 16) at scale j = 3 
%          18 shearing filters (16 by 16) at scale j = 4 (fine scale) 



% Originally written by Glenn R. Easley on Feb 2, 2006.
% Modified by Wang-Q Lim, Dec. 2010
% Modified by Xavier Gibert-Serra, Feb 2013

for j = 1:length(num)
    n1 = m(j);
    level = num(j);
    
    % generate indexing coordinate for Pseudo-Polar Grid
    [x11,y11,x12,y12,F1]=gen_x_y_cordinates(n1);

    wf=windowing(ones(2*n1,1),2^level);
    w_s{j}=zeros(n1,n1,2^level); %initialize window array
    for k=1:2^level,
        temp=wf(:,k)*ones(n1,1)';
        w_s{j}(:,:,k)=rec_from_pol(temp,n1,x11,y11,x12,y12,F1); % convert window array into Cartesian coord.
        w_s{j}(:,:,k)=real(fftshift(ifft2(fftshift(w_s{j}(:,:,k)))))./sqrt(n1); 
    end
end

% Save filters to file
fileName = 'shearFilters';
for j = 1:length(num)
    fileName = [fileName sprintf('_%d',num(j))];
end
fileName = [fileName  '_' dataType '.bin'];

fout = fopen(fileName, 'wb');
% Save header
fwrite(fout, 'SHFM0002', 'char*1');     % Shearing filter Myer version 2 file
% Save number of scales
fwrite(fout, length(num), 'uint16');
% Save data type
if( strcmp(dataType,'single') )
    fwrite(fout, 0, 'uint16');              % 0 for single real, 2 for double real
else
    fwrite(fout, 2, 'uint16');              % 0 for single real, 2 for double real
end
% Save all scales
for j=1:length(num)
    % Filter dimension
    fwrite(fout, m(j), 'uint16');
    % Save number of directions
    fwrite(fout, size(w_s{j}, 3), 'uint16');
    % Save filter data
    for k=1:size(w_s{j},3)
        fwrite(fout, w_s{j}(:,:,k), dataType);
    end
end
% Save atrous filters
[f{1}, f{2}, f{3}, f{4}] = atrousfilters('maxflat');
for j=1:4
    % Filter lenghth
    fwrite(fout, length(f{j}), 'uint16');
    % Filter elements
    fwrite(fout, f{j}, dataType);
end

fclose(fout);
