% Demo routine for separating points and curves

% Based on code written by Wang-Q Lim on May 5, 2010. 
% Copyright 2010 by Wang-Q Lim. All Right Reserved.

% Modified by Xavier Gibert-Serra on May 14, 2013 to run on GPU
% Copyright (C) 2012-2013 University of Maryland. All rights reserved.


N = 5*150; % the number of random points
sigma = 20; % noise level
do_display = 0;
do_profile = 0;
shear_timers(3);

% Discover device capabilities
device_count = 0;
[status, device_count] = cudaGetDeviceCount(device_count);

% generate test image (points + curves) with noise
[img,nimg] = curve_point_512(N,sigma);

% CPU double is our gold standard
% generate shearing filters across scales j. 
shear_double=shearing_filters_Myer([80 80 80 80],[3 3 4 4],512);

% generate shearing filters across scales j. 
shear_single=shearing_filters_Myer([80 80 80 80],[3 3 4 4],512,'single');

% Check if we can do GPU single and GPU double
support_GPUsingle = device_count > 0;
support_GPUdouble = device_count > 0;

if support_GPUsingle
    fprintf('Free GPU memory = %d\n', uint64(GPUmem));
    try
        shear_GPUsingle=shearing_filters_Myer([80 80 80 80],[3 3 4 4],512,'GPUsingle');
    catch err
        support_GPUsingle = false;
        support_GPUdouble = false;
        disp('GPU single is not supported');
    end
end

if support_GPUdouble
    try
        shear_GPUdouble=shearing_filters_Myer([80 80 80 80],[3 3 4 4],512,'GPUdouble');
    catch err
        support_GPUdouble = false;
    end
end

% Benchmark CPU double
if ~exist('C_double','var')
    fprintf('Running CPU double\n');
    if do_profile, profile on, end
    % separation using shearlets and wavelets
    [C_double, P_double] = separate(nimg,4,10,3,3,[.1 .1 1.5 1.5],do_display,shear_double);
    if do_profile, profsave(profile('info'),'profile_double'); profile off, end
end

% Benchmark CPU single
if ~exist('C_single','var')
    nimg_single = single(nimg);
    fprintf('Running CPU single\n');
    if do_profile, profile on, end
    % separation using shearlets and wavelets
    [C_single, P_single] = separate(nimg_single,4,10,3,3,[.1 .1 1.5 1.5],do_display,shear_single);
    if do_profile, profsave(profile('info'),'profile_single'); profile off, end
end

% Benchmark GPU single
if support_GPUsingle
    nimg_gpu = GPUsingle(nimg);
    GPUsync;
    fprintf('Running GPU single\n');
    if do_profile, profile on, end
    [C_GPUsingle, P_GPUsingle] = separate(nimg_gpu,4,10,3,3,[.1 .1 1.5 1.5],do_display,shear_GPUsingle);
    if do_profile, profsave(profile('info'),'profile_GPUsingle'); profile off, end
    clear_shearing_filters_cuda(shear_GPUsingle);
    clear shear_GPUsingle
    clear nimg_gpu
    shear_timers(0);    % Display timers
    shear_timers(1);    % Reset timers
end

% Benchmark GPU double
if support_GPUdouble
    nimg_gpu = GPUdouble(nimg);
    GPUsync;
    fprintf('Running GPU double\n');
    if do_profile, profile on, end
    [C_GPUdouble, P_GPUdouble] = separate(nimg_gpu,4,10,3,3,[.1 .1 1.5 1.5],do_display,shear_GPUdouble);
    if do_profile, profsave(profile('info'),'profile_GPUdouble'); profile off, end
    clear nimg_gpu
    clear_shearing_filters_cuda(shear_GPUdouble);
    clear shear_GPUdouble
    shear_timers(0);    % Display timers
    shear_timers(1);    % Reset timers    
end

% display results 
figure; clf; imagesc(img); axis equal; axis tight; colormap jet; 
title('original image');

figure; clf; imagesc(nimg); axis equal; axis tight; colormap jet; 
title('noisy image');
  
figure; clf; imagesc(C_double, [0 max(max(C_double))]); axis equal; axis tight; colormap jet; 
title('separated image : curves');

figure; clf; imagesc(P_double, [0 max(max(P_double))]); axis equal; axis tight; colormap jet; 
title('separated image : points');

axis off; 

% Calculate errors
v_gold = [C_double(:); P_double(:)];
v_test = [C_single(:); P_single(:)];
fprintf('RMSE for single = %g\n', (mean((v_test - v_gold).^2)/mean((v_gold).^2))^.5);
if support_GPUsingle
    v_test = [single(C_GPUsingle(:)); single(P_GPUsingle(:))];
    fprintf('RMSE for GPU single = %g\n', (mean((v_test - v_gold).^2)/mean((v_gold).^2))^.5);
    clear C_GPUsingle P_GPUsingle
end
if support_GPUdouble
    v_test = [single(C_GPUdouble(:)); single(P_GPUdouble(:))];
    fprintf('RMSE for GPU double = %g\n', (mean((v_test - v_gold).^2)/mean((v_gold).^2))^.5);
    clear C_GPUdouble P_GPUdouble
end

% Clean up GPU resources
clear C_GPUsingle P_GPUsingle C_GPUdouble P_GPUdouble
fprintf('Free memory = %d\n', uint64(GPUmem));