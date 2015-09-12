function [im_overlay] = crack_overlay(labels, gt_crack, im)

missed = uint8(~labels & gt_crack);
im = uint8(im*256);
im_overlay = zeros([size(im) 3], 'uint8');
im_overlay(:,:,1) = im .* uint8(~missed);
im_overlay(:,:,2) = im .* uint8(~labels | missed);
im_overlay(:,:,3) = im .* uint8(~labels | missed);
