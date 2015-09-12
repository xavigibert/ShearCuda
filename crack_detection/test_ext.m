% Select input images
shear_f=shearing_filters_Myer([68 80 80 80],[3 3 3 4],512,'GPUsingle');
pfilt = 'maxflat';
shearNorm = com_norm(pfilt,[512 512],shear_f);
load shear_dirs.mat

d = dir('../extended_crack_data/');
image_name = {};
for idx=1:length(d)
    if ~d(idx).isdir & strcmpi(d(idx).name(end-3:end),'.pgm')
        image_name{end+1} = d(idx).name(1:end-4);
    end
end

results.gt_crack_area = zeros(length(image_name),1);
results.crack_hist = zeros(length(image_name),256);

for image_num = 1:length(image_name)
    
name_crack_img = sprintf('../extended_crack_data/%s.pgm',image_name{image_num});
name_crack_gt = sprintf('../extended_crack_data/%s_gtmask.png',image_name{image_num});
name_crack_enh = sprintf('../extended_results/%s_enh.png',image_name{image_num});
name_crack_crv = sprintf('../extended_results/%s_crv.png',image_name{image_num});
name_crack_tex = sprintf('../extended_results/%s_tex.png',image_name{image_num});
name_plot_roc = sprintf('../extended_results/%s_roc.eps',image_name{image_num});
name_det_int = sprintf('../extended_results/%s_det_int.png',image_name{image_num});
name_det_rec = sprintf('../extended_results/%s_det_rec.png',image_name{image_num});
name_det_sh = sprintf('../extended_results/%s_det_sh.png',image_name{image_num});
% name_det_gc = sprintf('../extended_results/%s_det_gc.png',image_name{image_num});
 name_ovr_gt = sprintf('../extended_results/%s_ovr_gt.png',image_name{image_num});

%yrange = 46:478;
%yrange = 43:402;
yrange = 1:512;
xrange = 1:512;

% Debugging parameters
display = 0;
verbose = 0;
if ~usejava('Desktop')
    display = 0;
end

% Processing parameters
param.numIter = 23;
param.deltaMax = 90;
param.stop = 1.5;
param.gamma = 2;
param.coeff = [0.3 0.25 0.65 1.1];
param.useDCT = 0;
param.wavelet_name = 'Symmlet';
param.wavelet_val = 5;
design_pd = 0.9;        % Design prob detection

% Clustering parametersname_crack_gt
clust.w = [ 1.20 0.75 0.00 0.00 ];
clust.gamma = 900;
clust.lambda_rel = 0.1;  % Ratio between data term and smoothness term
clust.offset_rel = 3;    % Sensitivity

lambda = clust.lambda_rel * clust.gamma;
bias = clust.offset_rel * lambda;

img = imread(name_crack_img);
gt_labels = imread(name_crack_gt);
gt = (gt_labels(:,:,1) == 128 & gt_labels(:,:,2) == 32 & gt_labels(:,:,3) == 32);
mask_fastener = ~(gt_labels(:,:,1) == 103 & gt_labels(:,:,2) == 163 & gt_labels(:,:,3) == 255) ...
    & ~(gt_labels(:,:,1) == 255 & gt_labels(:,:,2) == 255 & gt_labels(:,:,3) == 128) ...
    & ~(gt_labels(:,:,1) == 0 & gt_labels(:,:,2) == 0 & gt_labels(:,:,3) == 255);
% Disable normalization if there is a fastener
do_normalize = nnz(mask_fastener) == 0;
mask_bg = ~bwmorph(gt,'dilate',3) & mask_fastener;
gt_tiny = (gt_labels(:,:,1) == 128 & gt_labels(:,:,2) == 128 & gt_labels(:,:,3) == 255);
results.gt_crack_area(image_num) = nnz(gt | gt_tiny);
gt = bwmorph(gt,'skel',inf) | gt_tiny;
mask_bg = mask_bg(yrange,xrange);
mask_crack = gt(yrange,xrange) > 0;

% Process patch
if do_normalize
    nimg = normalizeHorIntensity(double(img));
else
    nimg = double(img);
end
% Normalize intensity
nimg = single(nimg * 130.0 / mean(nimg(:)));
img_disp = nimg/256*1.5;

img_gpu = GPUsingle(nimg);
[Cgpu,Pgpu,Rgpu,shearCt] = separateHairlineCrack(img_gpu,4,param,display,shear_f,0,shearNorm,verbose);

% Prepare low level features
shearAngles = cell(1,4);
for j=1:4, shearAngles{j} = GPUsingle(shear_dirs{j} * pi / 180); end
ll_features = llCrackFeatures(shearCt, shearAngles, clust.w);
%gc = gc_crack_affinities(ll_features, clust.gamma, lambda, bias);

P = double(Pgpu);
C = double(Cgpu);
R = double(Rgpu);

im1 = img_disp * 256;
im2 = (R+P) * 256;
im2 = (im2/median(im2(:))*130/256*2-.25) * 256;
im3 = ((C/median(C(:))*130)/256*2-.25) * 256;
% Shearlet space features
im4 = single(ll_features(:,:,3)).^2 + single(ll_features(:,:,4)).^2 + ...
      single(ll_features(:,:,7)).^2 + single(ll_features(:,:,8)).^2;
im4 = 255 - 10 * log(1 + 128*im4);

imwrite(im1/256,name_crack_enh);
imwrite(im2/256,name_crack_tex);
imwrite(im3/256,name_crack_crv);

if display,
    figure(1), imshow(im1/256)
    figure(2), imshow(im2/256)
    figure(3), imshow(im3/256)
end

im1c = im1(yrange,xrange);
im3c = im3(yrange,xrange);
im4c = im4(yrange,xrange);

hist_c1 = hist(im1c(mask_crack),[0:255]);
pdf_c1 = hist_c1 / sum(hist_c1(:));
hist_b1 = hist(im1c(mask_bg),[0:255]);
pdf_b1 = hist_b1 / sum(hist_b1(:));

hist_c3 = hist(im3c(mask_crack),[0:255]);
pdf_c3 = hist_c3 / sum(hist_c3(:));
hist_b3 = hist(im3c(mask_bg),[0:255]);
pdf_b3 = hist_b3 / sum(hist_b3(:));

hist_c4 = hist(im4c(mask_crack),[0:255]);
pdf_c4 = hist_c4 / sum(hist_c4(:));
hist_b4 = hist(im4c(mask_bg),[0:255]);
pdf_b4 = hist_b4 / sum(hist_b4(:));

results.bg_hist(image_num,:) = hist_b4;
results.c_hist(image_num,:) = hist_c4;
results.hist(image_num,:) = hist(im4c(mask_fastener),[0:255]);

if display,
    figure(4), plot([0:255], pdf_c1, 'r-', [0:255], pdf_b1, 'g-', 'LineWidth', 1.5);
    figure(5), plot([0:255], pdf_c3, 'r-', [0:255], pdf_b3, 'g-', 'LineWidth', 1.5);
    figure(6), plot([0:255], pdf_c4, 'r-', [0:255], pdf_b4, 'g-', 'LineWidth', 1.5);
end

pd1 = cumsum(pdf_c1);
pf1 = cumsum(pdf_b1);
pd3 = cumsum(pdf_c3);
pf3 = cumsum(pdf_b3);
pd4 = cumsum(pdf_c4);
pf4 = cumsum(pdf_b4);
% pd5 = [];
% pf5 = [];

% Run graphcuts with different offset so we can get an ROC curve
% saved_crack_gc = false;
count_mask_crack = sum(mask_crack(:));
count_mask_bg = sum(mask_bg(:));
% for offset_rel = -1:0.1:25,
%     bias = offset_rel * lambda;
%     labels1 = cracks_gc(gc, lambda, bias);
%     labels = labels1(yrange,xrange);
%     pd5(end+1) = sum(labels(mask_crack)) / count_mask_crack;
%     pf5(end+1) = sum(labels(mask_bg)) / count_mask_bg;
%     if pd5(end) < design_pd && ~saved_crack_gc
%         fprintf('Graphcut: offset_rel = %.1f, PD = %.3f. PFA = %.3f\n', offset_rel, pd5(end), pf5(end));
%         imwrite(crack_overlay(labels1, gt, img_disp), name_det_gc);
%         saved_crack_gc = true;
%     end
% end
% pd5 = fliplr([1 pd5 0]);
% pf5 = fliplr([1 pf5 0]);

% Save overlaid images
% imwrite(crack_overlay(im1<find(pd1>=design_pd,1), gt, img_disp), name_det_int);
% imwrite(crack_overlay(im3<find(pd3>=design_pd,1), gt, img_disp), name_det_rec);
% imwrite(crack_overlay(im4<find(pd4>=design_pd,1), gt, img_disp), name_det_sh);
imwrite(crack_overlay(im1<find(pf1>=1e-3,1), gt, img_disp), name_det_int);
imwrite(crack_overlay(im3<find(pf3>=1e-3,1), gt, img_disp), name_det_rec);
imwrite(crack_overlay(im4<find(pf4>=1e-3,1), gt, img_disp), name_det_sh);
imwrite(crack_overlay(gt, gt, img_disp), name_ovr_gt);

figure(1),
semilogx(pf1, pd1, 'b-', pf3, pd3, 'r-', pf4, pd4, 'g-', 'LineWidth', 1.5);
xlabel('False positive rate');
ylabel('True positive rate');
axis([0.95e-5 1 0 1]);
grid on
legend('intensity original','intensity reconstructed','shearlet', ...
    'Location', 'SouthEast');
p = get(gcf, 'Position');
%set(gcf, 'Position', [1400 640 560 308]);
set(gcf, 'Position', [p(1) p(2) 560 308]);
set(gca, 'Position', [0.08 0.15 0.89 0.815]);

%figure(2)
%imshow(img_disp);

figure(3)
plot(results.hist(image_num,1:254));

prec1 = 1 ./ (1+(pf1./pd1)*count_mask_bg/count_mask_crack);
prec3 = 1 ./ (1+(pf3./pd3)*count_mask_bg/count_mask_crack);
prec4 = 1 ./ (1+(pf4./pd4)*count_mask_bg/count_mask_crack);
% prec5 = 1 ./ (1+(pf5./pd5)*count_mask_bg/count_mask_crack);
% figure(9), plot(prec1, pd1, 'b-', prec3, pd3, 'r-', prec4, pd4, 'g-', prec5, pd5, 'm-');
% xlabel('Precision');
% ylabel('Recall');
% axis([0 1 0.65 1]);
% grid on
% legend('intensity original','intensity reconstructed','shearlet','shearlet+graphcut');
% set(gcf, 'Position', [1400 640 560 308]);
% set(gca, 'Position', [0.08 0.11 0.89 0.815]);

dice1 = 2 * (prec1 .* pd1) ./ (prec1 + pd1);
dice3 = 2 * (prec3 .* pd3) ./ (prec3 + pd3);
dice4 = 2 * (prec4 .* pd4) ./ (prec4 + pd4);
%dice5 = 2 * (prec5 .* pd5) ./ (prec5 + pd5);

fprintf('Intensity AUC=%.5f F1=%.5f PD(PF=1e-3)=%.4f(th=%5.1f) PD(PF=1e-4)=%.4f(th=%5.1f)\n', ...
    trapz(pf1,pd1), max(dice1), pd_from_pfa(1e-3, pd1, pf1), th_from_pfa(1e-3, 0:255, pf1), ...
    pd_from_pfa(1e-4, pd1, pf1), th_from_pfa(1e-4, 0:255, pf1) );
fprintf('Int Rec   AUC=%.5f F1=%.5f PD(PF=1e-3)=%.4f(th=%5.1f) PD(PF=1e-4)=%.4f(th=%5.1f)\n', ...
    trapz(pf3,pd3), max(dice3), pd_from_pfa(1e-3, pd3, pf3), th_from_pfa(1e-3, 0:255, pf3), ...
pd_from_pfa(1e-4, pd3, pf3), th_from_pfa(1e-4, 0:255, pf3) );
fprintf('Shearlet  AUC=%.5f F1=%.5f PD(PF=1e-3)=%.4f(th=%5.1f) PD(PF=1e-4)=%.4f(th=%5.1f)\n', ...
    trapz(pf4,pd4), max(dice4), pd_from_pfa(1e-3, pd4, pf4), th_from_pfa(1e-3, 0:255, pf4), ...
    pd_from_pfa(1e-4, pd4, pf4), th_from_pfa(1e-4, 0:255, pf4) );
% fprintf('Graphcut  AUC=%.5f F1=%.5f PD(PF=1e-3)=%.4f(th=%5.1f) PD(PF=1e-4)=%.4f(th=%5.1f)\n', ...
%     trapz(pf5,pd5), max(dice5), pd_from_pfa(1e-3, pd5, pf5), th_from_pfa(1e-3, 0:255, pf5), ...
%     pd_from_pfa(1e-4, pd5, pf5), th_from_pfa(1e-4, 0:255, pf5) );

end

clear_shearing_filters_cuda(shear_f);
clear shear_f


% Analyze results
save results.mat results image_name
