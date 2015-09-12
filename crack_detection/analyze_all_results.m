load results.mat

% best_auc = 0;
% best_pd = [];
% best_pf = [];
% 
% for ref_quantile = [0:0.0001:0.002 0.003:0.001:0.01 0.02:0.01:0.1 0.15:0.05:1]
%     for th_off = -35:25
%         for gamma = [0:0.05:0.5 0:-0.05:-1];
%             [pd,pf,auc,ll] = analyze_results(results,th_off,ref_quantile,gamma);
%             if auc > best_auc
%                 best_auc = auc;
%                 best_pd = pd;
%                 best_pf = pf;
%                 best_ref_quantile = ref_quantile;
%                 best_th_off = th_off;
%                 best_gamma = gamma;
%                 best_ll = ll;
%             end
%         end
%     end
% end
% Show best ROC
figure(1), plot(best_pf,best_pd,'Linewidth',2), grid on
xlabel('False positive rate');
ylabel('True positive rate');

% Generate best/worst images
num_images = length(results.gt_crack_area);
num_cracks = nnz(results.gt_crack_area>0);
num_no_cracks = nnz(results.gt_crack_area==0);
gt = zeros(1,num_images);
for idx = 1:num_images
    gt(idx) = results.gt_crack_area(idx) > 0;
end
ll_true = best_ll(gt == 1);
idx_true = find(gt == 1);
ll_false = best_ll(gt == 0);
idx_false = find(gt == 0);
[ll_sorted_true,idx] = sort(ll_true);
idx_true_sorted = idx_true(idx);
[ll_sorted_false,idx] = sort(-ll_false);
idx_false_sorted = idx_false(idx);

pos_ex = zeros(256*4+3, 256*4+3);
neg_ex = zeros(256*4+3, 256*4+3);
for idx = 1:16
    image_num = idx_true_sorted(end-idx+1);
    name_crack_img = sprintf('../extended_crack_data/%s.pgm',image_name{image_num});
    res_img = imresize(im2double(imread(name_crack_img)), 0.5);
    pos_ex(floor((idx-1)/4)*257+1:floor((idx-1)/4)*257+256,mod(idx-1,4)*257+1:mod(idx-1,4)*257+256) = res_img;
end
for idx = 1:16
    image_num = idx_false_sorted(end-idx+1);
    name_crack_img = sprintf('../extended_crack_data/%s.pgm',image_name{image_num});
    res_img = imresize(im2double(imread(name_crack_img)), 0.5);
    neg_ex(floor((idx-1)/4)*257+1:floor((idx-1)/4)*257+256,mod(idx-1,4)*257+1:mod(idx-1,4)*257+256) = res_img;
end

missed_img = zeros(512*2+1, 512*2+1);
false_img = zeros(512*2+1, 512*2+1);
for idx = 1:4
    image_num = idx_true_sorted(idx);
    name_crack_img = sprintf('../extended_crack_data/%s.pgm',image_name{image_num});
    missed_img(floor((idx-1)/2)*513+1:floor((idx-1)/2)*513+512,mod(idx-1,2)*513+1:mod(idx-1,2)*513+512) = im2double(imread(name_crack_img));
end
for idx = 1:4
    image_num = idx_false_sorted(idx);
    name_crack_img = sprintf('../extended_crack_data/%s.pgm',image_name{image_num});
    false_img(floor((idx-1)/2)*513+1:floor((idx-1)/2)*513+512,mod(idx-1,2)*513+1:mod(idx-1,2)*513+512) = im2double(imread(name_crack_img));
end
figure(3), imshow(pos_ex), title('Positive examples');
figure(4), imshow(neg_ex), title('Negative examples');

figure(5), imshow(false_img), title('False positives (type I errors)');
figure(6), imshow(missed_img), title('False negatives (type II errors)');
