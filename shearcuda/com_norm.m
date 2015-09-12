function E = com_norm(pfilt,n,w_s)
%Compute the norm of shearlets for each scale and direction. 

%Input 
%pfilt : name of laplacian pyramid filter.
%n : size of input image (e.g : 256 X 256, 512 X 512...)
%w_s : cell array of directional shearing filters


%NOTE) For more details on the parameters 'pfilt','w_s', and 'opt'.
%      see instruction.txt.

%Output 
%E : l^2 norm of shearlets across scales and directions



% Written by Wang-Q Lim on May 5, 2010. 
% Copyright 2010 by Wang-Q Lim. All Right Reserved.

% Modified by Xavier Gibert <gibert@umiacs.umd.edu> on Jan 17, 2013
% to support GPU acceleration
% Copyright 2013 University of Maryland

    % Check data type
    dataType = type_of_shear_dict(w_s);

    switch dataType
        case 'double'
            F = ones(n(1),n(2));
        case 'single'
            F = ones(n(1),n(2),'single');
        case 'GPUsingle'
            F = GPUsingle(ones(n(1),n(2),'single'));
        case 'GPUdouble'
            F = GPUdouble(ones(n(1),n(2)));
    end
    X = fftshift(real(ifft2(F))) * sqrt(prod(size(F)));
    if strcmp(dataType,'single') || strcmp(dataType,'double')
        C=shear_trans(X,pfilt,w_s);
    else
        C=shear_trans_cuda(X,pfilt,w_s);
    end
    
    % Compute norm of shearlets (exact)
    for s=1:length(C)
        for w=1:size(C{s},3)
            if strcmp(dataType,'single') || strcmp(dataType,'double')
                A = C{s}(:,:,w);
                E(s,w) = real(double(sqrt(sum(sum(A.*conj(A))) / prod(size(A)))));
            else
                E(s,w) = norm_cuda(C{s},2,w) / sqrt(size(C{s},1)*size(C{s},2));
            end
        end
    end
