% Select input images
shear_f=shearing_filters_Myer([68 80 80 80],[3 3 3 4],512,'GPUsingle');
pfilt = 'maxflat';
shearNorm = com_norm(pfilt,[512 512],shear_f);
load shear_dirs.mat

for image_num = 1:3;
% Input image
name_crack_img = sprintf('../crack_data/crack%d.png',image_num);
% Crack ground truth image
name_crack_gt = sprintf('../crack_data/crack%d_gt.png',image_num);
% Canny contour ground truth image
name_crack_gt_cn = sprintf('../crack_data/crack%d_gt_cn.png',image_num);
% Output images
name_crack_enh = sprintf('../results/crack%d_enh.png',image_num);
name_crack_crv = sprintf('../results/crack%d_crv.png',image_num);
name_crack_tex = sprintf('../results/crack%d_tex.png',image_num);
name_plot_roc = sprintf('../results/crack%d_roc.eps',image_num);
name_det_int = sprintf('../results/crack%d_det_int.png',image_num);
name_det_rec = sprintf('../results/crack%d_det_rec.png',image_num);
name_det_sh = sprintf('../results/crack%d_det_sh.png',image_num);
name_det_cn = sprintf('../results/crack%d_det_cn.png',image_num);
name_ovr_gt = sprintf('../results/crack%d_ovr_gt.png',image_num);

%yrange = 46:478;
yrange = 40:472;
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
gt = imread(name_crack_gt);
gt_cn = imread(name_crack_gt_cn);

% Process patch
nimg = normalizeHorIntensity(double(img));
% Normalize intensity
nimg = single(nimg * 130.0 / mean(nimg(:)));
img_disp = nimg/256*1.5;

img_gpu = GPUsingle(nimg);
[Cgpu,Pgpu,Rgpu,shearCt] = separateHairlineCrack(img_gpu,4,param,display,shear_f,0,shearNorm,verbose);

% Prepare low level features
shearAngles = cell(1,4);
for j=1:4, shearAngles{j} = GPUsingle(shear_dirs{j} * pi / 180); end
ll_features = llCrackFeatures(shearCt, shearAngles, clust.w);

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

system(['convert ' name_crack_enh ' ' name_crack_enh(1:end-3) 'eps']);
system(['convert ' name_crack_tex ' ' name_crack_tex(1:end-3) 'eps']);
system(['convert ' name_crack_crv ' ' name_crack_crv(1:end-3) 'eps']);

if display,
    figure(1), imshow(im1/256)
    figure(2), imshow(im2/256)
    figure(3), imshow(im3/256)
end

im1c = im1(yrange,xrange);
im3c = im3(yrange,xrange);
im4c = im4(yrange,xrange);
mask_bg = ~bwmorph(gt,'dilate',4);
mask_bg_canny = mask_bg & ~bwmorph(gt_cn,'dilate',4);
mask_bg = mask_bg(yrange,xrange);
mask_bg_canny = mask_bg_canny(yrange,xrange);
mask_crack = gt(yrange,xrange) > 0;
mask_canny = gt_cn(yrange,xrange) > 0;

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

if display,
    figure(4), plot([0:255], pdf_c1, 'r-', [0:255], pdf_b1, 'g-', 'LineWidth', 1.5);
    figure(5), plot([0:255], pdf_c3, 'r-', [0:255], pdf_b3, 'g-', 'LineWidth', 1.5);
    figure(6), plot([0:255], pdf_c4, 'r-', [0:255], pdf_b4, 'g-', 'LineWidth', 1.5);
end

pd1 = [zeros(1,255) 1];
pf1 = [zeros(1,255) 1];
for i=2:255
    [pd1(i), pf1(i)] = ecm(im1c < i, mask_bg, mask_crack, 1);
end

pd3 = [zeros(1,255) 1];
pf3 = [zeros(1,255) 1];
for i=2:255
    [pd3(i), pf3(i)] = ecm(im3c < i, mask_bg, mask_crack, 1);
end

pd4 = [zeros(1,255) 1];
pf4 = [zeros(1,255) 1];
for i=2:255
    [pd4(i), pf4(i)] = ecm(im4c < i, mask_bg, mask_crack, 1);
end

pd5 = [zeros(1,127) 1];
pf5 = [zeros(1,127) 1];
for i=2:127
    [pd5(i), pf5(i)] = ecm(edge(im1c,'canny',(128-i)/128), mask_bg_canny, mask_canny, 1);
end

count_mask_crack = sum(mask_crack(:));
count_mask_bg = sum(mask_bg(:));

figure, semilogx( pf4, pd4, 'g-', pf3, pd3, 'r-', pf1, pd1, 'b-', pf5, pd5, 'm-', 'LineWidth', 2);
xlabel('False positive rate');
ylabel('True positive rate');
axis([0.95e-5 1 0 1]);
grid on
legend('Shearlet-C','Shearlet-I','Intensity','Canny', ...
    'Location', 'SouthEast');
set(gcf, 'Position', [1400 640 560 308]);
set(gca, 'Position', [0.08 0.15 0.89 0.815]);

prec1 = 1 ./ (1+(pf1./pd1)*count_mask_bg/count_mask_crack);
prec3 = 1 ./ (1+(pf3./pd3)*count_mask_bg/count_mask_crack);
prec4 = 1 ./ (1+(pf4./pd4)*count_mask_bg/count_mask_crack);
prec5 = 1 ./ (1+(pf5./pd5)*count_mask_bg/count_mask_crack);

dice1 = 2 * (prec1 .* pd1) ./ (prec1 + pd1);
dice3 = 2 * (prec3 .* pd3) ./ (prec3 + pd3);
dice4 = 2 * (prec4 .* pd4) ./ (prec4 + pd4);
dice5 = 2 * (prec5 .* pd5) ./ (prec5 + pd5);

fprintf('==== Image %d ====\n',image_num);
fprintf('Shearlet-C AUC=%.5f F1=%.5f PD(PF=1e-3)=%.4f(th=%5.1f) PD(PF=1e-4)=%.4f(th=%5.1f)\n', ...
    trapz(pf4,pd4), max(dice4), pd_from_pfa(1e-3, pd4, pf4), th_from_pfa(1e-3, 0:255, pf4), ...
    pd_from_pfa(1e-4, pd4, pf4), th_from_pfa(1e-4, 0:255, pf4) );
fprintf('Shearlet-I AUC=%.5f F1=%.5f PD(PF=1e-3)=%.4f(th=%5.1f) PD(PF=1e-4)=%.4f(th=%5.1f)\n', ...
    trapz(pf3,pd3), max(dice3), pd_from_pfa(1e-3, pd3, pf3), th_from_pfa(1e-3, 0:255, pf3), ...
    pd_from_pfa(1e-4, pd3, pf3), th_from_pfa(1e-4, 0:255, pf3) );
fprintf('Intensity  AUC=%.5f F1=%.5f PD(PF=1e-3)=%.4f(th=%5.1f) PD(PF=1e-4)=%.4f(th=%5.1f)\n', ...
    trapz(pf1,pd1), max(dice1), pd_from_pfa(1e-3, pd1, pf1), th_from_pfa(1e-3, 0:255, pf1), ...
    pd_from_pfa(1e-4, pd1, pf1), th_from_pfa(1e-4, 0:255, pf1) );
fprintf('Canny      AUC=%.5f F1=%.5f PD(PF=1e-3)=%.4f(th=%5.1f) PD(PF=1e-4)=%.4f(th=%5.1f)\n', ...
    trapz(pf5,pd5), max(dice5), pd_from_pfa(1e-3, pd5, pf5), th_from_pfa(1e-3, 0:255, pf5), ...
    pd_from_pfa(1e-4, pd5, pf5), th_from_pfa(1e-4, 0:255, pf5) );

[f1, idx1] = max(2 * (prec1 .* pd1) ./ (4*prec1 + pd1));
[f3, idx3] = max(2 * (prec3 .* pd3) ./ (4*prec3 + pd3));
[f4, idx4] = max(2 * (prec4 .* pd4) ./ (4*prec4 + pd4));
[f5, idx5] = max(2 * (prec5 .* pd5) ./ (4*prec5 + pd5));

% Hide boundary
hide_mask = zeros(512);
hide_mask(yrange, xrange) = 1;

% Save overlaid images
imwrite(crack_overlay(im1<idx1 & hide_mask, gt & hide_mask, img_disp), name_det_int);
imwrite(crack_overlay(im3<idx3 & hide_mask, gt & hide_mask, img_disp), name_det_rec);
imwrite(crack_overlay(im4<idx4 & hide_mask, gt & hide_mask, img_disp), name_det_sh);
imwrite(crack_overlay(edge(im1,'canny',(128-idx5)/128) & hide_mask, gt_cn & hide_mask, img_disp), name_det_cn);
imwrite(crack_overlay(gt & hide_mask, gt & hide_mask, img_disp), name_ovr_gt);

system(['convert ' name_det_int ' ' name_det_int(1:end-3) 'eps']);
system(['convert ' name_det_rec ' ' name_det_rec(1:end-3) 'eps']);
system(['convert ' name_det_sh ' ' name_det_sh(1:end-3) 'eps']);
system(['convert ' name_det_cn ' ' name_det_cn(1:end-3) 'eps']);
system(['convert ' name_ovr_gt ' ' name_ovr_gt(1:end-3) 'eps']);

end

clear_shearing_filters_cuda(shear_f);
clear shear_f
