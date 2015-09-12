function test_forloop1

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end
%I = sqrt(-1);
I = 0;

gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;
txtfun = GPUtest.txtfun;

%% test
for f=1:length(gpufun)
for N = [150];
  GPUtestLOG(sprintf('********** Testing forloop1 (%s) **********', txtfun{f}),0);
  GPUtestLOG(sprintf('* kernel_fft_rot (%d,%d,24,100)',N,N),0);
  GPUtestLOG(sprintf('* vis_fft        (%d,%d,24)',N,N),0);
  GPUtestLOG(sprintf('* C              (%d,%d,24)',N,N),0);
  
  %%
%   kernel_fft_rot = feval(gpufun{f},rand(N,N,24,100)+I*rand(N,N,24,100));
%   vis_fft = feval(gpufun{f},rand(N,N,24)+I*rand(N,N,24));
  kernel_fft_rot = feval(gpufun{f},rand(N,N,24,100));
  vis_fft = feval(gpufun{f},rand(N,N,24));
  
  C = zeros(N,N,24,feval(gpufun{f}));
  tic
  forloop1(kernel_fft_rot, vis_fft, C);
  t1 = toc;
  GPUtestLOG(sprintf('* Optimized code time     = %f',t1),0);
  
  
  %%
  C1 = zeros(N,N,24,feval(gpufun{f}));
  tic
  for i=1:size(kernel_fft_rot,4)
    for j=1:size(vis_fft,3)
      out= real(ifft2(vis_fft(:,:,j).*kernel_fft_rot(:,:,j,i)));
      C1(:,:,j)= C1(:,:,j) + out;
    end
  end
  t2= toc;
  compareCPUGPU(feval(cpufun{f},C1), C);
  GPUtestLOG(sprintf('* Non optimized code time = %f',t2),0);
  
  
  %%
%   kernel_fft_rot = feval(cpufun{f},rand(N,N,24,100)+I*rand(N,N,24,100));
%   vis_fft = feval(cpufun{f},rand(N,N,24)+I*rand(N,N,24));
  kernel_fft_rot = feval(cpufun{f},rand(N,N,24,100));
  vis_fft = feval(cpufun{f},rand(N,N,24));
  
  C1 = zeros(N,N,24,txtfun{f});
  tic
  for i=1:size(kernel_fft_rot,4)
    for j=1:size(vis_fft,3)
      out= real(ifft2(vis_fft(:,:,j).*kernel_fft_rot(:,:,j,i)));
      C1(:,:,j)= C1(:,:,j) + out;
    end
  end
  t3=toc;
  GPUtestLOG(sprintf('* CPU time                = %f',t3),0);
  
end
end
end

