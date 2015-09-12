function test_subsasgn

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end
%%
GPUtestLOG('***********************************************',0);

if (GPUtest.checkCompiler==1)
  GPUtestLOG('**** Compilation mode not supported',1);
end

type = GPUtest.type;

if (type == 1)
  rangei = [0 sqrt(-1)];
  rangej = [0 sqrt(-1)];
end

if (type == 2)
  rangei = 0;
  rangej = 0;
end

if (type == 3)
  rangei = [sqrt(-1)];
  rangej = [sqrt(-1)];
end

gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;

txtfun = GPUtest.txtfun;


for f=1:length(cpufun)
  for c=1:length(cpufun)
    
    for i=rangei
      for j=rangej
        
        if (i==0)
          type1 = 'real';
        end
        if (i==sqrt(-1))
          type1 = 'complex';
        end
        
        if (j==0)
          type2 = 'real';
        end
        if (j==sqrt(-1))
          type2 = 'complex';
        end
        
        GPUtestLOG(sprintf('**** Testing SUBSASGN (%s,%s) (%s,%s) ',txtfun{f},txtfun{c},type1,type2),0);
        
        switch txtfun{f}
          case 'single'
            NMAXF = 4100;
          case 'double'
            NMAXF = 4100/2;
          otherwise
            NMAXF = 4100;
        end
        
        switch txtfun{c}
          case 'single'
            NMAXC = 4100;
          case 'double'
            NMAXC = 4100/2;
          otherwise
            NMAXC = 4100;
        end
        
        NMAXF = min([NMAXF NMAXC]);
        
        
        %% Scalars
        % Test RHS scalar
        % Scalars are automatically converted from CPU to GPU.
        
        A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
        B = feval(gpufun{c},rand(1,1)+i*rand(1,1));
        Ah = feval(cpufun{f},A);
        Bh = feval(cpufun{c},B);
        
        A(1:2:10,:) = B;
        Ah(1:2:10,:) = Bh;
        
        compareCPUGPU(Ah,A);
        
        % Same test using CPU scalar
        A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
        B = feval(cpufun{c},rand(1,1)+i*rand(1,1));
        Ah = feval(cpufun{f},A);
        Bh = feval(cpufun{c},B);
        
        A(1:2:10,:) = B;
        Ah(1:2:10,:) = Bh;
        
        compareCPUGPU(Ah,A);
        
        %%
        A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
        B = feval(gpufun{c},rand(2000,2000)+j*rand(2000,2000));
        Ah = feval(cpufun{f},A);
        Bh = feval(cpufun{c},B);
        
        A(300:-10:200,400:-1:300) = B(300:-10:200,400:-1:300);
        Ah(300:-10:200,400:-1:300) = Bh(300:-10:200,400:-1:300);
        
        clear B
        clear Bh
        
        compareCPUGPU(Ah,A);
        
        if (GPUtest.fastMode==0)
          %%
          B = feval(gpufun{c},rand(2000,2000)+j*rand(2000,2000));
          A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
          
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(250:300,400:-1:350) = B(250:300,400:-1:350);
          Ah(250:300,400:-1:350) = Bh(250:300,400:-1:350);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
          B = feval(gpufun{c},rand(2000,2000)+j*rand(2000,2000));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(:) = B(:);
          Ah(:) = Bh(:);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          
          %%
          A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
          B = feval(gpufun{c},rand(2000,2000)+j*rand(2000,2000));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:2:2000,1:2:2000) = B(1:1000,1:1000);
          Ah(1:2:2000,1:2:2000) = Bh(1:1000,1:1000);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
          B = feval(gpufun{c},rand(2000,2000)+j*rand(2000,2000));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:2:2000) = B(1:1000);
          Ah(1:2:2000) = Bh(1:1000);
          clear B
          clear Bh
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(NMAXF,NMAXF));
          B = feval(gpufun{c},rand(NMAXF,NMAXF));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1000:2000,:) = B(1000:2000,:);
          Ah(1000:2000,:) = Bh(1000:2000,:);
          clear B
          clear Bh
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
          B = feval(gpufun{c},rand(10,10)+j*rand(10,10));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(10:-1:7) = B(1:4);
          Ah(10:-1:7) = Bh(1:4);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
          B = feval(gpufun{c},rand(10,10)+j*rand(10,10));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:2:8) = B(1:4);
          Ah(1:2:8) = Bh(1:4);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
          B = feval(gpufun{c},rand(10,10)+j*rand(10,10));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:2:8) = B(1:4);
          Ah(1:2:8) = Bh(1:4);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
          B = feval(gpufun{c},rand(10,10)+j*rand(10,10));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A([1 2 4 5],:) = B(1:4,:);
          Ah([1 2 4 5],:) = Bh(1:4,:);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          %%
          A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
          B = feval(gpufun{c},rand(10,10)+j*rand(10,10));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(10:-1:7,:) = B(1:4,:);
          Ah(10:-1:7,:) = Bh(1:4,:);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
          B = feval(gpufun{c},rand(2100,2100)+j*rand(2100,2100));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(2000:-1:1000,:) = B(1000:2000,:);
          Ah(2000:-1:1000,:) = Bh(1000:2000,:);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          clear A;
          clear B;
          A = feval(gpufun{f},rand(NMAXF,NMAXF));
          B = feval(gpufun{c},rand(NMAXF,NMAXF));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:end) = B(1:end);
          Ah(1:end) = Bh(1:end);
          clear B
          clear Bh
          compareCPUGPU(Ah,A);
          
          %%
          clear A;
          clear B;
          A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
          B = feval(gpufun{c},rand(2100,2100)+j*rand(2100,2100));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:end) = B(1:end);
          Ah(1:end) = Bh(1:end);
          clear B
          clear Bh
          compareCPUGPU(Ah,A);
          %%
          clear A;
          clear B;
          A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
          B = feval(gpufun{c},rand(2100,2100)+j*rand(2100,2100));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1,1:end) = B(1,1:end);
          Ah(1,1:end) = Bh(1,1:end);
          clear B
          clear Bh
          compareCPUGPU(Ah,A);
          
          %%
          clear A;
          clear B;
          A = feval(gpufun{f},rand(NMAXF,NMAXF));
          B = feval(gpufun{c},rand(NMAXF,NMAXF));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1,1:end) = B(1,1:end);
          Ah(1,1:end) = Bh(1,1:end);
          clear B
          clear Bh
          compareCPUGPU(Ah,A);
          
          %%
          clear A;
          clear B;
          A = feval(gpufun{f},rand(NMAXF,NMAXF));
          B = feval(gpufun{c},rand(NMAXF,NMAXF));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          for kk=1:floor((numel(A)-300)/10):(numel(A)-300)
            offset = kk;
            A((1+offset):(256+offset)) = B((1+offset):(256+offset));
            Ah((1+offset):(256+offset)) = Bh((1+offset):(256+offset));
            compareCPUGPU(Ah,A);
            
          end
          %%
          A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
          B = feval(gpufun{c},rand(2100,2100)+j*rand(2100,2100));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          for kk=1:floor((numel(A)-300)/10):(numel(A)-300)
            offset = kk;
            A((1+offset):(256+offset)) = B((1+offset):(256+offset));
            Ah((1+offset):(256+offset)) = Bh((1+offset):(256+offset));
            compareCPUGPU(Ah,A);
            
          end
          %%
          A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:2,1:2) = B(1:2,1:2);
          Ah(1:2,1:2) = Bh(1:2,1:2);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1,1:10) = B(1:10);
          Ah(1,1:10) = Bh(1:10);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          tic;A(1:2,4,1:4,1) = B(1:2,1:4);toc;
          tic;Ah(1:2,4,1:4,1) = Bh(1:2,1:4);toc;
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(900,500)+i*rand(900,500));
          B = feval(gpufun{c},rand(300,500)+j*rand(300,500));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          tic;A(1:300,1:500) = B;toc;
          tic;Ah(1:300,1:500) = Bh;toc;
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          
          %%
          A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:10) = B(1:10);
          Ah(1:10) = Bh(1:10);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1:numel(A)) = B(1:numel(A));
          Ah(1:numel(A)) = Bh(1:numel(A));
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(1,3,4,3) = B(1,2:2,6:6);
          Ah(1,3,4,3) = Bh(1,2:2,6:6);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(:,:) = B(:,:);
          Ah(:,:) = Bh(:,:);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          A(:,1:end,1) = B(:,1:end,1);
          Ah(:,1:end,1) = Bh(:,1:end,1);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
          %%
          A = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          idx = colon(1,3,feval(gpufun{f}));
          idxh = feval(cpufun{f},idx);
          A(idx,1:end,1) = B(idx,1:end,1);
          Ah(idxh,1:end,1) = Bh(idxh,1:end,1);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          %%
          A = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
          B = feval(gpufun{c},rand(3,4,6)+j*rand(3,4,6));
          Ah = feval(cpufun{f},A);
          Bh = feval(cpufun{c},B);
          
          idx = colon(1,3,feval(gpufun{f}));
          idxh = feval(cpufun{f},idx);
          A(idx,idx,1) = B(idx,idx,1);
          Ah(idxh,idxh,1) = Bh(idxh,idxh,1);
          
          clear B
          clear Bh
          
          compareCPUGPU(Ah,A);
          
        end
        
      end
    end
  end
  GPUtestLOG('***********************************************',0);
end
