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
%
% For example, save_shearing_filters([30 30 36 36],[3 3 4 4],'single');
% produces cell array 'dshear' consisting of 
%          10 shearing filters (30 by 30) at scale j = 1 (coarse scale)  
%          10 shearing filters (30 by 30) at scale j = 2 
%          18 shearing filters (36 by 36) at scale j = 3 
%          18 shearing filters (36 by 36) at scale j = 4 (fine scale) 



% Originally written by Glenn R. Easley on Feb 2, 2006.
% Modified by Wang-Q Lim, Dec. 2010
% Modified by Xavier Gibert-Serra, Feb 2013

for j = 1:length(num)
    n1 = m(j); level = num(j);    
    [x11,y11,x12,y12,F1]=gen_x_y_cordinates(n1);
    N=2*n1;
    M=2^level+2;

    wf=windowing(ones(N,1),2^level,1);
    w_s{j}=zeros(n1,n1,M);
    w = zeros(n1,n1);
    for k=1:M,
        temp=wf(:,k)*ones(N/2,1)';
        w_s{j}(:,:,k)=rec_from_pol(temp,n1,x11,y11,x12,y12,F1);
        w = w + w_s{j}(:,:,k);
    end
    for k = 1:M
        w_s{j}(:,:,k) = sqrt(1./w.*w_s{j}(:,:,k));
        w_s{j}(:,:,k) = real(fftshift(ifft2(ifftshift((w_s{j}(:,:,k))))));
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
fwrite(fout, 'SHFM0001', 'char*1');     % Shearing filter Myer version 1 file
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
    fwrite(fout, w_s{j}, dataType);
end
% Save atrous filters
[f{1}, f{2}, f{3}, f{4}] = atrousfilters('maxflat','double');
for j=1:4
    % Filter lenghth
    fwrite(fout, length(f{j}), 'uint16');
    % Filter elements
    fwrite(fout, f{j}, dataType);
end

fclose(fout);
