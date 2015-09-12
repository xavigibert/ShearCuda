function test_assign

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

%%
GPUtestLOG('***********************************************',0);
gpufun = GPUtest.gpufun;
cpufun = GPUtest.cpufun;
txtfun = GPUtest.txtfun;

% GPUtest.type
% 1 - real/complex
% 2 - real
% 3 - complex

type = GPUtest.type;
switch type
  case 1
    rangei = [0 sqrt(-1)];
    rangej = [0 sqrt(-1)];
  case 2
    rangei = 0;
    rangej = 0;
  case 3
    rangei = sqrt(-1);
    rangej = sqrt(-1);
end

testfun = GPUtest.testfun;

%% Test assign
for f=1:length(cpufun)
  
  
  
  for i=rangei
    switch i
      case 0
        complexity = 'real';
      case sqrt(-1)
        complexity = 'complex';
    end
    
    
    GPUtestLOG(sprintf('**** Testing assign (%s,%s)',txtfun{f}, complexity),0);
    
    %% Scalars
    % Test RHS scalar
    % Scalars are automatically converted from CPU to GPU. 
    
    A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
    B = feval(gpufun{f},rand(1,1)+i*rand(1,1));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1,A,B,[1,2,10],':');
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1,A,B,[1,2,10],':');
      
    end
    
    Ah(1:2:10,:) = Bh;
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
    B = feval(gpufun{f},rand(1,1)+i*rand(1,1));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      a = [1,10];
      b = 1;
      GPUcompileStart('comp_assign','-f','-verbose0',A,B,a,b)
      assign(1,A,B,a,b);
      GPUcompileStop
      comp_assign(A,B,[1,2,10],':');
    else
      assign(1,A,B,[1,2,10],':');
      
    end
    
    Ah(1:2:10,:) = Bh;
    compareCPUGPU(Ah,A);
    
    %% Same test using CPU scalar
    A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
    B = feval(cpufun{f},rand(1,1)+i*rand(1,1));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1,A,B,[1,2,10],':');
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1,A,B,[1,2,10],':');
    end
    Ah(1:2:10,:) = Bh;
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(1000,1000)+i*rand(1000,1000));
    B = feval(cpufun{f},rand(1,1)+i*rand(1,1));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      a = [1,10];
      b = 1;
      GPUcompileStart('comp_assign','-f','-verbose0',A,B,a,b)
      assign(1,A,B,a,b);
      GPUcompileStop
      comp_assign(A,B,[1,2,10],':');
    else
      assign(1,A,B,[1,2,10],':');
    end
    
    Ah(1:2:10,:) = Bh;
    compareCPUGPU(Ah,A);
    
    
    
    %%
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      B = B(300:-10:200,400:-1:300);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [300,-10,200],[400,-1,300]);
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1, A, B(300:-10:200,400:-1:300), [300,-10,200],[400,-1,300]);
    end
    
    Ah(300:-10:200,400:-1:300) = Bh(300:-10:200,400:-1:300);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      B = B(300:-10:200,400:-1:300);
      a = 1;
      b = 2;
      GPUcompileStart('comp_assign','-f','-verbose0',A,B,a,b)
      assign(1, A, B, a, b);
      GPUcompileStop
      comp_assign(A,B, [300,-10,200],[400,-1,300] );
      
    else
      assign(1, A, B(300:-10:200,400:-1:300), [300,-10,200],[400,-1,300]);
    end
    
    Ah(300:-10:200,400:-1:300) = Bh(300:-10:200,400:-1:300);
    compareCPUGPU(Ah,A);
    
    %%
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      B = B(250:300,400:-1:350);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [250,1,300],[400,-1,350]);
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1, A, B(250:300,400:-1:350), [250,1,300],[400,-1,350]);
    end
    Ah(250:300,400:-1:350) = Bh(250:300,400:-1:350);
    compareCPUGPU(Ah,A);
    
    %%
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    if (GPUtest.checkCompiler==1)
      B = B(250:300,400:-1:350);
      a = 1;
      b = 2;
      GPUcompileStart('comp_assign','-f','-verbose0',A,B,a,b)
      assign(1, A, B, a, b);
      GPUcompileStop
      comp_assign(A,B, [250,1,300],[400,-1,350]);
      
    else
      assign(1, A, B(250:300,400:-1:350), [250,1,300],[400,-1,350]);
    end
    Ah(250:300,400:-1:350) = Bh(250:300,400:-1:350);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(:) = B(:);
    if (GPUtest.checkCompiler==1)
      B=B(:);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, ':');
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1,A,B(:),':');
    end
    Ah(:) = Bh(:);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(:) = B(:);
    if (GPUtest.checkCompiler==1)
      B=B(:);
      a = 1;
      GPUcompileStart('comp_assign','-f','-verbose0',A,B,a)
      assign(1, A, B, a);
      GPUcompileStop
      comp_assign(A,B,':');
      
     else
      assign(1,A,B(:),':');
    end
    Ah(:) = Bh(:);
    compareCPUGPU(Ah,A);
    
    
    %%
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:2:2000,1:2:2000) = B(1:1000,1:1000);
    if (GPUtest.checkCompiler==1)
      B=B(1:1000,1:1000);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1,2,2000],[1,2,2000]);
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1,A,B(1:1000,1:1000),[1,2,2000],[1,2,2000]);
    end
    Ah(1:2:2000,1:2:2000) = Bh(1:1000,1:1000);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:2:2000,1:2:2000) = B(1:1000,1:1000);
    if (GPUtest.checkCompiler==1)
      B=B(1:1000,1:1000);
      a=2;
      b=1;
      GPUcompileStart('comp_assign','-f','-verbose0',A,B,a,b)
      assign(1, A, B, a, b);
      GPUcompileStop
      comp_assign(A,B,[1,2,2000],[1,2,2000]);
      
    else
      assign(1,A,B(1:1000,1:1000),[1,2,2000],[1,2,2000]);
    end
    Ah(1:2:2000,1:2:2000) = Bh(1:1000,1:1000);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    B = feval(gpufun{f},rand(2000,2000)+i*rand(2000,2000));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:2:2000) = B(1:1000);
    if (GPUtest.checkCompiler==1)
      B=B(1:1000);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1,2,2000]);
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1,A,B(1:1000),[1,2,2000]);
    end
    Ah(1:2:2000) = Bh(1:1000);
    compareCPUGPU(Ah,A);
    
    %%
    if strcmp(txtfun{f},'single')
      A = feval(gpufun{f},rand(4100,4100));
      B = feval(gpufun{f},rand(4100,4100));
    else
      A = feval(gpufun{f},rand(3100,3100));
      B = feval(gpufun{f},rand(3100,3100));
    end
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1000:2000,:) = B(1000:2000,:);
    if (GPUtest.checkCompiler==1)
      B=B(1000:2000,:);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1000,1,2000],':');
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1,A,B(1000:2000,:),[1000,1,2000],':');
    end
    Ah(1000:2000,:) = Bh(1000:2000,:);
    clear Bh;
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    B = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(10:-1:7) = B(1:4);
    if (GPUtest.checkCompiler==1)
      B=B(1:4);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [10,-1,7]);
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1, A, B(1:4), [10,-1,7]);
    end
    Ah(10:-1:7) = Bh(1:4);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    B = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:2:8) = B(1:4);
    if (GPUtest.checkCompiler==1)
      B=B(1:4);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1,2,8]);
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1, A, B(1:4), [1,2,8]);
    end
    Ah(1:2:8) = Bh(1:4);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    B = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A([1 2 4 5],:) = B(1:4,:);
    if (GPUtest.checkCompiler==1)
      B=B(1:4,:);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, {[1 2 4 5]},':');
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1,A,B(1:4,:),{[1 2 4 5]},':');
    end
    Ah([1 2 4 5],:) = Bh(1:4,:);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    B = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A([1 2 4 5],:) = B(1:4,:);
    if (GPUtest.checkCompiler==1)
      B=B(1:4,:);
      a = 1;
      GPUcompileStart('comp_assign','-f','-verbose0',A,B,a)
      assign(1, A, B, a,':');
      GPUcompileStop
      comp_assign(A,B, {[1 2 4 5]});
    else
      assign(1,A,B(1:4,:),{[1 2 4 5]},':');
    end
    Ah([1 2 4 5],:) = Bh(1:4,:);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    B = feval(gpufun{f},rand(10,10)+i*rand(10,10));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(10:-1:7,:) = B(1:4,:);
    if (GPUtest.checkCompiler==1)
      B=B(1:4,:);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [10,-1,7],':');
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1,A,B(1:4,:),[10,-1,7],':');
    end
    Ah(10:-1:7,:) = Bh(1:4,:);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    B = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(2000:-1:1000,:) = B(1000:2000,:);
    if (GPUtest.checkCompiler==1)
      B=B(1000:2000,:);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B,  [2000,-1,1000], ':' );
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1, A, B(1000:2000,:),  [2000,-1,1000], ':' );
    end
    Ah(2000:-1:1000,:) = Bh(1000:2000,:);
    compareCPUGPU(Ah,A);
    
    %%
    clear A;
    clear B;
    if strcmp(txtfun{f},'single')
      A = feval(gpufun{f},rand(4100,4100));
      B = feval(gpufun{f},rand(4100,4100));
    else
      A = feval(gpufun{f},rand(3100,3100));
      B = feval(gpufun{f},rand(3100,3100));
    end
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:end) = B(1:end);
    if (GPUtest.checkCompiler==1)
      B=B(1:end);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1,1,END]);
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1, A, B(1:end), [1,1,END]);
    end
    Ah(1:end) = Bh(1:end);
    compareCPUGPU(Ah,A);
    
    %%
    clear A;
    clear B;
    A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    B = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:end) = B(1:end);
    if (GPUtest.checkCompiler==1)
      B=B(1:end);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1,1,END]);
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1, A, B(1:end), [1,1,END]);
    end
    Ah(1:end) = Bh(1:end);
    compareCPUGPU(Ah,A);
    
    %%
    clear A;
    clear B;
    A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    B = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:end) = B(1:end);
    if (GPUtest.checkCompiler==1)
      B=B(1:end-10);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1,1,END-10]);
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1, A, B(1:end-10), [1,1,END-10]);
    end
    Ah(1:end-10) = Bh(1:end-10);
    compareCPUGPU(Ah,A);
    
    %%
    clear A;
    clear B;
    A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    B = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1,1:end) = B(1,1:end);
    if (GPUtest.checkCompiler==1)
      B=B(1,1:end);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, 1 ,[1,1,END] );
      GPUcompileStop
      comp_assign(A,B);
      
    else
      assign(1, A, B(1,1:end), 1 ,[1,1,END] );
    end
    Ah(1,1:end) = Bh(1,1:end);
    compareCPUGPU(Ah,A);
    
    %%
    clear A;
    clear B;
    if strcmp(txtfun{f},'single')
      A = feval(gpufun{f},rand(4100,4100));
      B = feval(gpufun{f},rand(4100,4100));
    else
      A = feval(gpufun{f},rand(3100,3100));
      B = feval(gpufun{f},rand(3100,3100));
    end
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1,1:end) = B(1,1:end);
    if (GPUtest.checkCompiler==1)
      B=B(1,1:end);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, 1 ,[1,1,END]);
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1, A, B(1,1:end), 1 ,[1,1,END]);
    end
    Ah(1,1:end) = Bh(1,1:end);
    compareCPUGPU(Ah,A);
    
    %%
    clear A;
    clear B;
    if strcmp(txtfun{f},'single')
      A = feval(gpufun{f},rand(4100,4100));
      B = feval(gpufun{f},rand(4100,4100));
    else
      A = feval(gpufun{f},rand(3100,3100));
      B = feval(gpufun{f},rand(3100,3100));
    end
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    for kk=1:floor((numel(A)-300)/10):(numel(A)-300)
      offset = kk;
      %A((1+offset):(256+offset)) = B((1+offset):(256+offset));
      if (GPUtest.checkCompiler==1)
        C=B((1+offset):(256+offset));
        GPUcompileStart('comp_assign','-f','-verbose0',A,C)
        assign(1, A, C, [(1+offset),1,(256+offset)]);
        GPUcompileStop
        comp_assign(A,C);
      else
        assign(1, A, B((1+offset):(256+offset)),  [(1+offset),1,(256+offset)]);
      end
      Ah((1+offset):(256+offset)) = Bh((1+offset):(256+offset));
      compareCPUGPU(Ah,A);
      
    end
    
    
    %%
    A = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    B = feval(gpufun{f},rand(2100,2100)+i*rand(2100,2100));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    for kk=1:floor((numel(A)-300)/10):(numel(A)-300)
      offset = kk;
      %A((1+offset):(256+offset)) = B((1+offset):(256+offset));
      if (GPUtest.checkCompiler==1)
        C=B((1+offset):(256+offset));
        GPUcompileStart('comp_assign','-f','-verbose0',A,C)
        assign(1, A, C, [(1+offset),1,(256+offset)]);
        GPUcompileStop
        comp_assign(A,C);
      else
        assign(1, A, B((1+offset):(256+offset)),[(1+offset),1,(256+offset)]);
      end
      Ah((1+offset):(256+offset)) = Bh((1+offset):(256+offset));
      compareCPUGPU(Ah,A);
      
    end
    
    %%
    A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
    B = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:2,1:2) = B(1:2,1:2);
    if (GPUtest.checkCompiler==1)
        B=B(1:2,1:2);
        GPUcompileStart('comp_assign','-f','-verbose0',A,B)
        assign(1, A, B, [1,1,2], [1,1,2]);
        GPUcompileStop
        comp_assign(A,B);
    else
      assign(1, A, B(1:2,1:2), [1,1,2], [1,1,2]);
    end
    Ah(1:2,1:2) = Bh(1:2,1:2);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
    B = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1,1:10) = B(1:10);
    if (GPUtest.checkCompiler==1)
        B=B(1:10);
        GPUcompileStart('comp_assign','-f','-verbose0',A,B)
        assign(1, A, B, 1, [1,1,10]);
        GPUcompileStop
        comp_assign(A,B);
    else
      assign(1,A, B(1:10), 1, [1,1,10]);
    end
    Ah(1,1:10) = Bh(1:10);
    
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
    B = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:2,4,1:4,1) = B(1:2,1:4);
    if (GPUtest.checkCompiler==1)
        B=B(1:2,1:4);
        GPUcompileStart('comp_assign','-f','-verbose0',A,B)
        assign(1, A, B, [1,1,2], 4, [1,1,4], 1);
        GPUcompileStop
        comp_assign(A,B);
    else
      assign(1 ,A, B(1:2,1:4), [1,1,2], 4, [1,1,4], 1);
    end
    Ah(1:2,4,1:4,1) = Bh(1:2,1:4);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(900,500)+i*rand(900,500));
    B = feval(gpufun{f},rand(300,500)+i*rand(300,500));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:300,1:500) = B;
    if (GPUtest.checkCompiler==1)
        GPUcompileStart('comp_assign','-f','-verbose0',A,B)
        assign(1, A, B, [1,1,300], [1,1,500]);
        GPUcompileStop
        comp_assign(A,B);
    else
      assign(1, A, B, [1,1,300], [1,1,500]);
    end
    Ah(1:300,1:500) = Bh;
    compareCPUGPU(Ah,A);
    
    
    %%
    A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
    B = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:10) = B(1:10);
    if (GPUtest.checkCompiler==1)
      B = B(1:10);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1,1,10]);
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1, A, B(1:10), [1,1,10]);
    end
    Ah(1:10) = Bh(1:10);
    
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    B = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1:numel(A)) = B(1:numel(A));
    if (GPUtest.checkCompiler==1)
      B = B(1:numel(A));
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, [1,1,numel(A)]);
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1, A, B(1:numel(A)), [1,1,numel(A)]);
    end
    Ah(1:numel(A)) = Bh(1:numel(A));
    
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(3,4,5,3)+i*rand(3,4,5,3));
    B = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(1,3,4,3) = B(1,2:2,6:6);
    if (GPUtest.checkCompiler==1)
      B = B(1,2:2,6:6);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, 1,3,4,3);
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1, A, B(1,2:2,6:6), 1,3,4,3);
    end
    Ah(1,3,4,3) = Bh(1,2:2,6:6);
    
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    B = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(:,:) = B(:,:);
    if (GPUtest.checkCompiler==1)
      B = B(:,:);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B,  ':',':');
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1, A, B(:,:), ':',':');
    end
    Ah(:,:) = Bh(:,:);
    compareCPUGPU(Ah,A);
    
    %%
    A = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    B = feval(gpufun{f},rand(3,4,6)+i*rand(3,4,6));
    Ah = feval(cpufun{f},A);
    Bh = feval(cpufun{f},B);
    
    %A(:,1:end,1) = B(:,1:end,1);
    if (GPUtest.checkCompiler==1)
      B = B(:,1:end,1);
      GPUcompileStart('comp_assign','-f','-verbose0',A,B)
      assign(1, A, B, ':', [1,1,END], 1);
      GPUcompileStop
      comp_assign(A,B);
    else
      assign(1, A, B(:,1:end,1), ':', [1,1,END], 1);
    end
    Ah(:,1:end,1) = Bh(:,1:end,1);
    
    compareCPUGPU(Ah,A);
    
   
    
    
    
  end
  
  
end
GPUtestLOG('***********************************************',0);
end
