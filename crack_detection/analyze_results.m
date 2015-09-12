function [pd,pf,auc,ll] = analyze_results(results,th_off,ref_quantile,gamma)

rng('default');

% Allocate
num_images = length(results.gt_crack_area);
num_cracks = nnz(results.gt_crack_area>0);
num_no_cracks = nnz(results.gt_crack_area==0);
ll = zeros(1,num_images);
gt = zeros(1,num_images);
pd = zeros(1,num_images);
pf = zeros(1,num_images);

% Run detector on all images
for idx = 1:num_images
    gt(idx) = results.gt_crack_area(idx) > 0;
    if ref_quantile > 0
        cdf = cumsum(results.hist(idx,:)/sum(results.hist(idx,:)));
        th = find(cdf >= ref_quantile, 1, 'first') + th_off;
        th_diff = th - 153;
        th = round(153 + gamma * th_diff);
    else
        th = 153 + th_off;
    end
%     if th < 125
%         th = 125;
%     elseif th > 200
%         th = 200;
%     end
    if th < 100
        th = 100;
    elseif th > 255
        th = 255;
    end
    ll(idx) = sum(results.hist(idx,1:th)) + rand(1)/2;
end
[y,idx] = sort(-ll);
pd = [0 cumsum(gt(idx))/num_cracks];
pf = [0 cumsum(1-gt(idx))/num_no_cracks];
auc = trapz(pf,pd);
