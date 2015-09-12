function [C, P, residual, Ct] = separateHairlineCrack(img,level2,param,display,shear,keepInterm,E,verbose)
% This routine performs image separation to separate two geometrically 
% different objects (points + curves).
% This routine builds up on MCA2_Bcr.m (in MCALab110 ) written by J. M. Fadili. 
%
%Output : P : separated image which mainly contains points.
%         C : separated image which mainly contains curves. 
%
%Input : 
%
%img : input image
%level2 : The number of levels for wavelet transform
%
%stop : stopping criterion ( eg. stop = 3 or 4)
%;
%itermax : the number of iterations of modified relaxation method.
%
%gamma: paramater for TV (soft-threshold) (e.g gamma = 3 or 4).
%
%coeff : coeff = [coeff(1)...coeff(L)] where 
%        L : the number of frequency bands.
%        coeff(j) : weight coefficient for each frequency band.
%        coeff(1) : weight for low frequecny 
%        ...
%        coeff(L) : weight for high frequency
%        ( see 're_weight_band.m'. )     
%
%display : display = 1 ---> display output of each iteration
%          
%shear : cell array of shearlet filters computed from shearing_filters_Myer.m
%      (see shearing_filters_Myer.m)
%
%keepInterm : keepInterm = 1 ---> keep itermediate data for each iteration

% Written by Wang-Q Lim on May 5, 2010. 
% Copyright 2010 by Wang-Q Lim. All Right Reserved.

% Modified by Xavier Gibert-Serra on June 28, 2012

n = length(img);
dataType = class(img);
isGPU = isa(img,'GPUsingle') || isa(img,'GPUdouble');
if isGPU, display = 0; end

% Settable parameters
itermax = param.numIter;
stop = param.stop;
gamma = param.gamma;
coeff = param.coeff;
userDeltaMax = param.deltaMax;

% Verbosity setting to view intermediate steps
if nargin < 8
    verbose = 1;
end

if nargin < 6
    keepInterm = 0;
end

if isfield(param,'wavelet_name')
    qmf = MakeONFilter(param.wavelet_name,param.wavelet_val,dataType);
else
    qmf = MakeONFilter('Symmlet',4,dataType);
end
qmf_haar = MakeONFilter('Haar',0,dataType);

pfilt = 'maxflat';
if verbose, tic; end
opt = [2 1];

% compute the L^2 norm of shearlets
if nargin < 7
    E = com_norm(pfilt,size(img),shear);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orig = img;
if isGPU
    img = re_weight_band_cuda(img,coeff,shear);
else
    img = re_weight_band(img,coeff);
end
%reconstruction with diffrent weight for each frequency band. 
%for example, coeff = [1 4 4 8 8] : weights ---> 1: low frequency... 8: high frequency. 

if userDeltaMax>0
    deltamax = userDeltaMax;
    sigma = 2.5;
    stop = stop*sigma;
else
    %estimate a starting thresholding parameter at each scale j.
    N = length(img);
    h1=MakeONFilter('Daubechies',4,dataType);

    % TO DO: Implement function FWT2_PO in CUDA
    %        Right now we perform the operation in CPU
    switch dataType
        case 'GPUsingle'
            %wc = FWT2_PO(single(img),log2(N)-1,single(h1));
            wc = mdwt(double(img), double(h1), 1);
        case 'GPUdouble'
            %wc = FWT2_PO(double(img),log2(N)-1,double(h1));
            wc = mdwt(double(img), double(h1), 1);
        case 'single'
            wc = mdwt(double(img), double(h1), 1);
        case 'double'
            %wc = FWT2_PO(img, log2(N)-1,h1);
            wc = mdwt(img, h1, 1);
    end
    hh = wc(N/2+1:N/2+floor(N/2),N/2+1:N/2+floor(N/2));hh = hh(:);
    sigma = MAD(hh);
    stop = stop*sigma;

    if isGPU
        Ct = shear_trans_cuda(img,pfilt,shear);
        [buff(1)] = thresh_cuda(Ct,0,2,E,ones(1,length(shear.filter)+1),1);
    else
        Ct = shear_trans(img,pfilt,shear);    
        [~, buff(1)] = thresh(Ct,0,2,E,ones(1,length(shear)+1),1);
    end

    %ylh = swt2(img,level2,'sym4');
    %[coeff buff(2)] = thresh1(ylh,0,1,E,ones(1,length(shear)+1),1);

    if isGPU
        [Ctw{1},Ctw{2}] = mrdwt_cuda(img,qmf,level2);
    else
        [Ctw{1},Ctw{2}] = mrdwt(img,qmf,level2);
    end

    buff(2) = max(abs(double(Ctw{2}(:))));
    deltamax = min(buff);
end

lambda=(deltamax/stop)^(1/(1-itermax)); % Exponential decrease.
delta = deltamax;

if isa(img,'GPUsingle')
    part{1} = zeros([n,n],GPUsingle);
    part{2} = zeros([n,n],GPUsingle);
    residual = zeros([n,n],GPUsingle);
    Ra = zeros([n,n],GPUsingle);
    %Preallocate Haar transform coefficients
    Cth{1} = zeros([n,n],GPUsingle);
    Cth{2} = zeros([n,n*3],GPUsingle);
elseif isa(img,'GPUdouble')
    part{1} = zeros([n,n],GPUdouble);
    part{2} = zeros([n,n],GPUdouble);
    residual = zeros([n,n],GPUdouble);
    Ra = zeros([n,n],GPUdouble);
    %Preallocate Haar transform coefficients
    Cth{1} = zeros([n,n],GPUdouble);
    Cth{2} = zeros([n,n*3],GPUdouble);
else
    part = zeros([n,n,2],dataType);
end

%if display
    %aviobj = avifile('neuron256.avi','compression','None','fps',2);
%screen_size=get(0,'screensize');
%            f1 = figure(1);
%            set(f1,'position',[0 0 screen_size(3),screen_size(4)]);
%end

if keepInterm
    % Prealllocate matrix for intermediate data
    C = zeros(size(img,1), size(img,2), itermax, 'single');
    P = zeros(size(img,1), size(img,2), itermax, 'single');
end

% Preallocate matrix for sparsity analysis
sparsity1 = zeros(itermax, 6);
sparsity2 = zeros(itermax, 2);

%Approximately solve l0 minimization at each scale j.
if verbose, tic2 = tic; end
for iter = 1:itermax
    if isGPU
        GPUminus(img, part{1}, residual);
        GPUminus(residual, part{2}, residual);
    else
        residual = img - (part(:,:,1)+part(:,:,2));
    end
    for k = 1:2
        if isGPU
            GPUplus(part{k},residual,Ra);
        else            
            Ra = part(:,:,k)+residual;
        end
        if k == 1
             %estimate curve part at jth level
             if isGPU
                 %apply the forward shearlet transform
                 if exist('Ct','var')
                     shear_trans_cuda(Ra,pfilt,shear,Ct);
                 else
                     Ct = shear_trans_cuda(Ra,pfilt,shear);
                 end
                 % in-place thresholding
                 thresh_cuda(Ct,(1.4)*delta,opt(1),E,ones(1,length(shear.filter)+1),0);
                 part{k} = inverse_shear_cuda(Ct,pfilt,shear);
                 %apply the inverse shearlet transform with thresholded coeff. 
                 part{k} = TVCorrection_cuda(part{k},gamma,qmf_haar,Cth);
                 %apply soft-thresholding with the (undecimated) Haar wavelet transform                  
             else
                 %apply the forward shearlet transform
                 Ct = shear_trans(Ra,pfilt,shear);
                 % regular thresholding
                 Ct = thresh(Ct,(1.4)*delta,opt(1),E,ones(1,length(shear)+1),0);
                 part(:,:,k) = inverse_shear(Ct,pfilt,shear);
                 %apply the inverse shearlet transform with thresholded coeff. 
                 part(:,:,k) = TVCorrection(part(:,:,k),gamma);
                 %apply soft-thresholding with the (undecimated) Haar wavelet transform                  
             end
        else
             if isGPU
                 if exist('Ctw','var')
                    mrdwt_cuda(Ra,qmf,level2,Ctw);
                 else
                    [Ctw{1},Ctw{2}] = mrdwt_cuda(Ra,qmf,level2);
                 end
                 % apply the undecimated wavelet transform
                 thresh_cuda(Ctw,1.4*delta,opt(2),E,0,0);
                 part{k} = mirdwt_cuda(Ctw{1},Ctw{2},qmf,level2);
                 % apply the inverse undecimated wavelet transform with
                 % thresholded coeff.
             else
                 [yl,yh] = mrdwt(Ra,qmf,level2);
                 % apply the undecimated wavelet transform
                 Ct{1} = yl; Ct{2} = yh;
                 Ct = thresh(Ct,1.4*delta,opt(2),E,0,0);
                 part(:,:,k) = mirdwt(Ct{1},Ct{2},qmf,level2);
                 % apply the inverse undecimated wavelet transform with
                 % thresholded coeff.
             end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%% display output %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if display,
            subplot(2,2,1);imagesc(orig);axis image; rmaxis;drawnow;
            colormap 'jet';
            c1 = max([max(max(sum(part,3))) 10]);
            subplot(2,2,2);imagesc(sum(part,3),[0 c1]);axis image;rmaxis;title('\Sigma_i Part_i');drawnow;
            colormap 'jet';
            for np=1:2
                c = max([max(max(part(:,:,np))) 10]);
                subplot(2,2,np+2);imagesc(part(:,:,np),[0 c]);axis image;rmaxis;title(sprintf('Part_%d',np));drawnow;
                colormap 'jet';
            end
            % Save in an AVI movie file.
	        %frame = getframe(f1);
            %aviobj = addframe(aviobj,frame);
	        %clear frame
        end
    end
    
    if verbose
        fprintf('ITER %d: TH=%4.1f, C has (%5d|%5d|%5d|%5d|%5d) coefs, P has (%5d|%5d) coefs\n', ...
            iter, delta, sparsity1(iter, 1), sparsity1(iter, 2), sparsity1(iter, 3), sparsity1(iter, 4), sparsity1(iter, 5), ...
            sparsity2(iter, 1), sparsity2(iter, 2));
    end
    
    delta=delta*lambda; % Updating thresholding parameter (Exponential decrease).
    
    if keepInterm
        % Save intermediate results
        if exist('uncorr_part1', 'var'), C(:,:,iter) = uncorr_part1; end
        P(:,:,iter) = part(:,:,2);
    end
end
if verbose
    fprintf('Main loop:'); toc(tic2);
end

% Find largest coefficients at each scale
% CoeffC = zeros(size(img,1), size(img,2), length(shear), 'single');
% CoeffC_min = zeros(size(img,1), size(img,2), length(shear), 'single');
% IdxC = zeros(size(img,1), size(img,2), length(shear), 'single');
% IdxC_min = zeros(size(img,1), size(img,2), length(shear), 'single');

% Extract features
% if ~ispc
% for j=1:length(shear)
%     Ct1_real = real(Ct1_unthresh{j+1});
%     % Normalize shearlet coefficients
%     for k=1:size(Ct1_real,3)
%         Ct1_real(:,:,k) = Ct1_real(:,:,k) / E(j+1,k);
%     end
%     % Find minimum and maximum response at each scale
%     [~, IdxC(:,:,j)] = max(abs(Ct1_real), [], 3);
%     [~, IdxC_min(:,:,j)] = min(abs(Ct1_real), [], 3);
%     for k=1:size(img,1)
%         for l=1:size(img,2)
%             CoeffC(k,l,j) = Ct1_real(k,l,IdxC(k,l,j));
%             CoeffC_min(k,l,j) = Ct1_real(k,l,IdxC_min(k,l,j));
%         end
%     end
% end
% end

% if ~keepInterm
%     C = part(:,:,1); P = part(:,:,2);
% end

if isGPU
    C = part{1}; P = part{2};
else
    C = part(:,:,1); P = part(:,:,2);
end

%aviobj = close(aviobj);
%close(f1);
if verbose
    if isGPU, GPUsync; end;
    fprintf('separate.m: '); toc(tic1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = TVCorrection(x,gamma)
% Total variation implemented using the approximate (exact in 1D) equivalence between the TV norm and the l_1 norm of the Haar (heaviside) coefficients.

qmf = single(MakeONFilter('Haar'));
[ll,wc] = mrdwt(x,qmf,1);
wc = SoftThresh(wc,gamma);
y = mirdwt(ll,wc,qmf,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = TVCorrection_cuda(x,gamma,qmf,Cth)
% Total variation implemented using the approximate (exact in 1D) equivalence between the TV norm and the l_1 norm of the Haar (heaviside) coefficients.

mrdwt_cuda(x,qmf,1,Cth);
softth_cuda(Cth{2},gamma);
y = mirdwt_cuda(Cth{1},Cth{2},qmf,1);

%wc = swt2(x,1,'haar');
%wc = SoftThresh(wc,gamma);
%y = iswt2(wc,'haar');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
