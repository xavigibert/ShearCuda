close all
shear_GPUsingle=shearing_filters_Myer([68 80 80 80],[3 3 3 4],256,'GPUsingle');
im = zeros(512);
im(254:257,1:512)=128;
im(255:256,1:512)=256;
angles = 0:0.5:179.5;
shear_dirs = cell(1,4);
shear_resp = cell(1,4);
angle_resp = cell(1,4);
for j = 1:4
  shear_dirs{j} = zeros(1,size(shear_GPUsingle.filter{j},3));
  shear_resp{j} = zeros(1,size(shear_GPUsingle.filter{j},3));
  angle_resp{j} = zeros(length(angles),size(shear_GPUsingle.filter{j},3));
end
for ang_idx = 1:length(angles);
  ang = angles(ang_idx);
  fprintf('ang=%d\n',ang);
  imr = imrotate(im,ang,'bicubic','crop');
  imr = GPUsingle(imr(129:384,129:384));
  Ct = shear_trans_cuda(imr,'maxflat',shear_GPUsingle);
  for j = 1:4,
    Ctj = single(Ct{j+1});
    for k = 1:size(Ctj,3),
      c = Ctj(:,:,k);
      nc = max(abs(c(:)));
      angle_resp{j}(ang_idx,k) = nc;
      if nc > shear_resp{j}(k)
        shear_dirs{j}(k) = ang;
        shear_resp{j}(k) = nc;
      end
    end
  end
end

%% Analyze responses
for j = 1:4,
    max_angle_resp = max(angle_resp{j},[],2);
    for k = 1:size(shear_GPUsingle.filter{j},3),
        vals = angles(angle_resp{j}(:,k) == max_angle_resp) - shear_dirs{j}(k);
        vals(vals > 90) = vals(vals > 90) - 180;
        vals(vals < -90) = vals(vals < -90) + 180;
        shear_dirs{j}(k) = shear_dirs{j}(k) + mean(vals);
        if shear_dirs{j}(k) > 180, shear_dirs{j}(k) = shear_dirs{j}(k) - 180; end
        if shear_dirs{j}(k) < 0, shear_dirs{j}(k) = shear_dirs{j}(k) + 180; end
        angle_resp{j}(:,k) =  angle_resp{j}(:,k) .* (angle_resp{j}(:,k) == max_angle_resp);
    end
end
ar = [angles, angles+180, 360] * pi / 180;
% Generate polar plots

%polar(ar,angle_resp{1});
for j = 1:4,
    figure(j)
    theta = [];
    rho = [];
    for k=1:size(shear_GPUsingle.filter{j},3),
        r = angle_resp{j}(:,k);
        theta = [theta; ar'];
        rho = [rho; angle_resp{j}(:,k); angle_resp{j}(:,k); angle_resp{j}(1,k)];
    end
    %polar(ar',[angle_resp{j}(:,k); angle_resp{j}(:,k); angle_resp{j}(1,k)]);
    polar(theta,rho);
    hold on;
    polar(shear_dirs{j}*pi/180, shear_resp{j},'r *');
    title(sprintf('Shearlet response (scale %d)',j));    
end

save shear_dirs.mat shear_dirs
