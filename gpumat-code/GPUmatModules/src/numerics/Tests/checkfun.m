function checkfun(op,config)
global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

%%
GPUtestLOG('***********************************************',0);

% Condition to cover (s = scalar). Each test done for each GPUtype (double,
% single, ...)

% For 2 operands
% A op B
% B op A
% A op s
% s op A

% A.' op B
% B op A.'
% A.' op B.'
% A.' op s
% s op A.'

% A' op B
% B op A'
% A' op B'
% A' op s
% s op A'

% For 1 operand
% op(A)
% op(A.')
% op(A')

% All of them for all possible combinations of real/complex numbers and different matrix dimensions (non only square matrices)

% If config.type>=0 override global GPUtest.type
if (config.type==-1)
  type = GPUtest.type;
else
  type = config.type;
end


% optype
% 1 - op(A)
% 2 - A op B
% 3 - op(A,B)
optype = config.optype;

% create A or B with elements transposed
trA = config.trA;
trB = config.trB;

% test function
if (~isa(config.testfun,'function_handle'))
  testfun = GPUtest.testfun;
else
  testfun = config.testfun;
end

K = single(0.4);
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

if (isa(config.gpufun,'double'))
  gpufun = GPUtest.gpufun;
else
  gpufun = config.gpufun;
end

if (isa(config.cpufun,'double'))
  cpufun = GPUtest.cpufun;
else
  cpufun = config.cpufun;
end

%gpufun = GPUtest.gpufun;
%cpufun = GPUtest.cpufun;

if (isa(config.txtfun,'double'))
  txtfun = GPUtest.txtfun;
else
  txtfun = config.txtfun;
end



if (GPUtest.checkCompiler==1)
  A = zeros(5,GPUsingle); % dummy
  B = zeros(5,GPUsingle); % dummy
  C = zeros(5,GPUsingle); % dummy
  switch optype
    case 1
      GPUcompileStart('comp_tmp','-f','-verbose0',A)
    case 2
      GPUcompileStart('comp_tmp','-f','-verbose0',A, B)
    case 3
      GPUcompileStart('comp_tmp','-f','-verbose0',A, B)
    case 4
      GPUcompileStart('comp_tmp','-f','-verbose0',A, B, C)
    otherwise
      error('Unexcpected option');
  end
  
  % run function
 switch optype
    case 1
      E = eval([op '(A)']);
    case 2
      E = eval(['(A)' op '(B)']);
    case 3
      eval(['GPU' op '(A, B)']);
    case 4
      eval(['GPU' op '(A, B, C)']);
     
  end
  
  switch optype
    case 1
      GPUcompileStop(E)
    case 2
      GPUcompileStop(E)
    case 3
      GPUcompileStop
    case 4
      GPUcompileStop
    otherwise
      error('Unexcpected option');
  end
  
  
  
end

%txtfun = GPUtest.txtfun;

%% Test operations with different types
if (optype==2 || optype==4)
  for f=1:length(cpufun)
    for i=0
      Am = feval(cpufun{f},feval(testfun,10)+i*feval(testfun,10));
      A = feval(gpufun{f},Am);
      
      % logical
      try
        B = 1;
        B = B>0;
        % filter = 1 is used when doing test of == and ~=. We some
        % elements to be the same
        if (config.filter==1)
          Am(1:3:end)= feval(cpufun{f},B);
          A = feval(gpufun{f},Am);
        end
        [E, err] = evaluateOP(op,optype,A,B);
        [Em, err] = evaluateOP(op,optype,Am,B);
        
        
        compareCPUGPU(Em,E);
        
        [E, err] = evaluateOP(op,optype,B,A);
        [Em, err] = evaluateOP(op,optype,B,Am);
        
        
        compareCPUGPU(Em,E);
        
        GPUtestLOG('here',1);
      catch
      end
      
      % double
      B = double(2);
      if (config.filter==1)
        Am(1:3:end)= feval(cpufun{f},B);
        A = feval(gpufun{f},Am);
      end
      
      [E,err] = evaluateOP(op,optype,A,B);
      [Em, err] = evaluateOP(op,optype,Am,B);
      
      compareCPUGPU(Em,E);
      
      [E,err] = evaluateOP(op,optype,B,A);
      [Em, err] = evaluateOP(op,optype,B,Am);
      
      compareCPUGPU(Em,E);
      
      % single
      B = single(2);
      if (config.filter==1)
        Am(1:3:end)= feval(cpufun{f},B);
        A = feval(gpufun{f},Am);
      end
      
      [E,err] = evaluateOP(op,optype,A,B);
      [Em, err] = evaluateOP(op,optype,Am,B);
      
      compareCPUGPU(Em,E);
      
      [E,err] = evaluateOP(op,optype,B,A);
      [Em, err] = evaluateOP(op,optype,B,Am);
      
      compareCPUGPU(Em,E);
      
      
      % int8 this currently gives error
      try
        B = int8(2);
        if (config.filter==1)
          Am(1:3:end)= feval(cpufun{f},B);
          A = feval(gpufun{f},Am);
        end
        
        [E,err] = evaluateOP(op,optype,A,B);
        [Em, err] = evaluateOP(op,optype,Am,B);
        
        compareCPUGPU(Em,E);
        
        [E,err] = evaluateOP(op,optype,B,A);
        [Em, err] = evaluateOP(op,optype,B,Am);
        
        compareCPUGPU(Em,E);
        
        GPUtestLOG('here',1);
      catch
      end
      
      % uint8 this currently gives error
      try
        B = uint8(2);
        if (config.filter==1)
          Am(1:3:end)= feval(cpufun{f},B);
          A = feval(gpufun{f},Am);
        end
        
        [E,err] = evaluateOP(op,optype,A,B);
        [Em, err] = evaluateOP(op,optype,Am,B);
        
        compareCPUGPU(Em,E);
        
        [E,err] = evaluateOP(op,optype,B,A);
        [Em, err] = evaluateOP(op,optype,B,Am);
        
        compareCPUGPU(Em,E);
        
        GPUtestLOG('here',1);
      catch
      end
      
      
      % int16 this currently gives error
      try
        B = int16(2);
        if (config.filter==1)
          Am(1:3:end)= feval(cpufun{f},B);
          A = feval(gpufun{f},Am);
        end
        
        [E,err] = evaluateOP(op,optype,A,B);
        [Em, err] = evaluateOP(op,optype,Am,B);
        
        compareCPUGPU(Em,E);
        
        [E,err] = evaluateOP(op,optype,B,A);
        [Em, err] = evaluateOP(op,optype,B,Am);
        
        compareCPUGPU(Em,E);
        
        GPUtestLOG('here',1);
      catch
      end
      
      
      % uint16 this currently gives error
      try
        B = uint16(2);
        if (config.filter==1)
          Am(1:3:end)= feval(cpufun{f},B);
          A = feval(gpufun{f},Am);
        end
        
        [E,err] = evaluateOP(op,optype,A,B);
        [Em, err] = evaluateOP(op,optype,Am,B);
        
        compareCPUGPU(Em,E);
        
        [E,err] = evaluateOP(op,optype,B,A);
        [Em, err] = evaluateOP(op,optype,B,Am);
        
        compareCPUGPU(Em,E);
        
        GPUtestLOG('here',1);
        
      catch
      end
      
      
      % int32 this currently gives error
      try
        B = int32(2);
        if (config.filter==1)
          Am(1:3:end)= feval(cpufun{f},B);
          A = feval(gpufun{f},Am);
        end
        
        [E, err] = evaluateOP(op,optype,A,B);
        [Em, err] = evaluateOP(op,optype,Am,B);
        
        compareCPUGPU(Em,E);
        
        [E, err] = evaluateOP(op,optype,B,A);
        [Em, err] = evaluateOP(op,optype,B,Am);
        
        compareCPUGPU(Em,E);
        
        GPUtestLOG('here',1);
      catch
      end
      
      
      % uint32 this currently gives error
      try
        B = uint32(2);
        if (config.filter==1)
          Am(1:3:end)= feval(cpufun{f},B);
          A = feval(gpufun{f},Am);
        end
        
        [E, err] = evaluateOP(op,optype,A,B);
        [Em, err] = evaluateOP(op,optype,Am,B);
        
        compareCPUGPU(Em,E);
        
        [E, err] = evaluateOP(op,optype,B,A);
        [Em, err] = evaluateOP(op,optype,B,Am);
        
        compareCPUGPU(Em,E);
        
        GPUtestLOG('here',1);
      catch
      end
      
      
    end
  end
  
end
%%
% Main test
% gpufun = GPUtest.gpufun;
% cpufun = GPUtest.cpufun;
% txtfun = GPUtest.txtfun;


if (optype==2 || optype==4)
  secondloop = length(cpufun);
end
if (optype==1 || optype==3)
  secondloop = 1;
end
%% Test operations with different types
for i1=1:length(cpufun)
  for j1=1:secondloop
    
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
        
        if (optype==2)
          GPUtestLOG(sprintf('**** Testing %s (%s,%s) (%s,%s) ',op,txtfun{i1},txtfun{j1},type1,type2),0);
        end
        
        if (optype==4)
          GPUtestLOG(sprintf('**** Testing GPU%s (%s,%s) (%s,%s) ',op,txtfun{i1},txtfun{j1},type1,type2),0);
        end
        
        if (optype==1 )
          GPUtestLOG(sprintf('**** Testing %s (%s) (%s) ',op,txtfun{i1},type1),0);
        end
        
        if (optype==3)
          GPUtestLOG(sprintf('**** Testing GPU%s (%s) (%s) ',op,txtfun{i1},type1),0);
        end
        
        % big kernel
        
        if (GPUtest.bigKernel==1)
          % filter = 1 is used when doing test of == and ~=. We some
          % elements to be the same
          % filter = 2 2 is used when testing logical
          % operators
          if (i==sqrt(-1))||(j==sqrt(-1))
            BIGN = 2000;
          else
            BIGN = 4100;
          end
          Am = feval(cpufun{i1},feval(testfun,BIGN,BIGN)+i*feval(testfun,BIGN,BIGN));
          
          if (config.filter==2)
            Am = Am>0;
          end
          
          
          if (optype==2 || optype==4)
            Bm = feval(cpufun{j1},feval(testfun,BIGN,BIGN)+j*feval(testfun,BIGN,BIGN));
            if (config.filter==1)
              Bm(1:2:end) = real(Am(1:2:end))+j*imag(Am(1:2:end));
            end
            if (config.filter==2)
              Bm = Bm>0;
            end
          else
            Bm = 0.0;
          end
          
          if (config.filter==1)
            Am(1:2:end) = real(Bm(1:2:end))+i*imag(Bm(1:2:end));
          end
          
          
          
          A = feval(gpufun{i1},Am);
          B = feval(gpufun{j1},Bm);
          
          [E, err] = evaluateOP(op,optype,A,B);
          [Em, err] = evaluateOP(op,optype,Am,Bm);
          
          clear A
          clear B
          clear Am
          clear Bm
          
          GPUmemClean;
          
          try
            compareCPUGPU(Em,E);
          catch
            err = lasterror;
            GPUtestLOG(err.message,1);
          end
          clear E
          clear Em
          
          
        end
        
        for M=GPUtest.M
          for N=GPUtest.N
            % trB and trA are used to test matrxi multiplication
            if (trA==0)
              Am = feval(cpufun{i1},feval(testfun,N,M)+i*feval(testfun,N,M));
            else
              Am = feval(cpufun{i1},feval(testfun,M,N)+i*feval(testfun,M,N));
            end
            if (config.filter==2)
              Am = Am>0;
            end
            
            if (optype==2 || optype==4)
              if (trB==0)
                Bm = feval(cpufun{j1},feval(testfun,N,M)+j*feval(testfun,N,M));
              else
                Bm = feval(cpufun{j1},feval(testfun,M,N)+j*feval(testfun,M,N));
              end
              if (config.filter==1)
                Bm(1:2:end) = real(Am(1:2:end))+j*imag(Am(1:2:end));
              end
              if (config.filter==2)
                Bm = Bm>0;
              end
            else
              Bm = 0.0;
            end
            
            if (config.filter==1)
              Am(1:2:end) = real(Bm(1:2:end))+i*imag(Bm(1:2:end));
            end
            
            A = feval(gpufun{i1},Am);
            B = feval(gpufun{j1},Bm);
            
            % A op B
            [E, err] = evaluateOP(op,optype,A,B);
            [Em, err] = evaluateOP(op,optype,Am,Bm);
            
            compareCPUGPU(Em,E);
            
            % B op A
            if (optype==2 || optype==4)
              [E, err] = evaluateOP(op,optype,B,A);
              [Em, err] = evaluateOP(op,optype,Bm,Am);
              
              compareCPUGPU(Em,E);
            end
            
            % A.' op B.'
            [E, err] = evaluateOP(op,optype,A.',B.');
            [Em, err] = evaluateOP(op,optype,Am.',Bm.');
            
            compareCPUGPU(Em,E);
            
            % B.' op A.'
            if (optype==2 || optype==4)
              [E, err] = evaluateOP(op,optype,B.',A.');
              [Em, err] = evaluateOP(op,optype,Bm.',Am.');
              
              compareCPUGPU(Em,E);
            end
            
            % A' op B'
            [E, err] = evaluateOP(op,optype,A',B');
            [Em, err] = evaluateOP(op,optype,Am',Bm');
            
            compareCPUGPU(Em,E);
            
            % B' op A'
            if (optype==2 || optype==4)
              [E, err] = evaluateOP(op,optype,B',A');
              [Em, err] = evaluateOP(op,optype,Bm',Am');
              
              compareCPUGPU(Em,E);
            end
            
            % transposed test
            if (trA==0)
              Am = feval(cpufun{i1},feval(testfun,N,M)+i*feval(testfun,N,M));
            else
              Am = feval(cpufun{i1},feval(testfun,M,N)+i*feval(testfun,M,N));
            end
            
            if (config.filter==2)
              Am = Am>0;
            end
            
            if (optype==2 || optype==4)
              if (trB==0)
                Bm = feval(cpufun{j1},feval(testfun,M,N)+j*feval(testfun,M,N));
              else
                Bm = feval(cpufun{j1},feval(testfun,N,M)+j*feval(testfun,N,M));
              end
              if (config.filter==1)
                Bm(1:2:end) = real(Am(1:2:end))+j*imag(Am(1:2:end));
              end
              if (config.filter==2)
                Bm = Bm>0;
              end
            else
              Bm = 0.0;
            end
            
            if (config.filter==1)
              Am(1:2:end) = real(Bm(1:2:end))+i*imag(Bm(1:2:end));
            end
            
            A = feval(gpufun{i1},Am);
            B = feval(gpufun{j1},Bm);
            
            % A.' op B
            [E, err] = evaluateOP(op,optype,A.',B);
            [Em, err] = evaluateOP(op,optype,Am.',Bm);
            
            compareCPUGPU(Em,E);
            
            % B op A.'
            if (optype==2 || optype==4)
              [E, err] = evaluateOP(op,optype,B,A.');
              [Em, err] = evaluateOP(op,optype,Bm,Am.');
              
              compareCPUGPU(Em,E);
            end
            
            
            
            %%%%
            % A' op B
            [E, err] = evaluateOP(op,optype,A',B);
            [Em, err] = evaluateOP(op,optype,Am',Bm);
            
            compareCPUGPU(Em,E);
            
            % B op A'
            if (optype==2 || optype==4)
              [E, err] = evaluateOP(op,optype,B,A');
              [Em, err] = evaluateOP(op,optype,Bm,Am');
              
              compareCPUGPU(Em,E);
            end
            
            
            
            if (optype==2 || optype==4)
              K = feval(cpufun{j1},0.1+j*0.07);
              if (config.filter==1)
                K = feval(cpufun{j1},real(Am(10)) + j*imag(Am(10)));
                Am(10) = real(K)+i*imag(K);
                A = feval(gpufun{i1},Am);
              end
              
              %% Scalar
              % A op K
              [E, err] = evaluateOP(op,optype,A,K);
              [Em, err] = evaluateOP(op,optype,Am,K);
              
              
              compareCPUGPU(Em,E);
              
              % K op A
              [E, err] = evaluateOP(op,optype,K,A);
              [Em, err] = evaluateOP(op,optype,K,Am);
              
              
              compareCPUGPU(Em,E);
              
              % A.' op K
              [E, err] = evaluateOP(op,optype,A.',K);
              [Em, err] = evaluateOP(op,optype,Am.',K);
              
              
              compareCPUGPU(Em,E);
              
              % K op A.'
              [E, err] = evaluateOP(op,optype,K,A.');
              [Em, err] = evaluateOP(op,optype,K,Am.');
              
              
              compareCPUGPU(Em,E);
              
              % A' op K
              [E, err] = evaluateOP(op,optype,A',K);
              [Em, err] = evaluateOP(op,optype,Am',K);
              
              compareCPUGPU(Em,E);
              
              % K op A'
              [E, err] = evaluateOP(op,optype,K,A');
              [Em, err] = evaluateOP(op,optype,K,Am');
              
              
              compareCPUGPU(Em,E);
              
              
              % A.' op K.'
              [E, err] = evaluateOP(op,optype,A.',K.');
              [Em, err] = evaluateOP(op,optype,Am.',K.');
              
              compareCPUGPU(Em,E);
              
              % K.' op A.'
              [E, err] = evaluateOP(op,optype,K.',A.');
              [Em, err] = evaluateOP(op,optype,K.',Am.');
              
              
              compareCPUGPU(Em,E);
              
              % A' op K'
              [E, err] = evaluateOP(op,optype,A',K');
              [Em, err] = evaluateOP(op,optype,Am',K');
              
              
              compareCPUGPU(Em,E);
              
              % K' op A'
              [E, err] = evaluateOP(op,optype,K',A');
              [Em, err] = evaluateOP(op,optype,K',Am');
              
              
              compareCPUGPU(Em,E);
              
              %% GPU scalar
              
              Km = feval(cpufun{j1},0.1+j*0.07);
              if (config.filter==1)
                Km = feval(cpufun{j1},real(Am(10)) + j*imag(Am(10)));
                Am(10) = real(K)+i*imag(K);
                A = feval(gpufun{i1},Am);
              end
              K  = feval(gpufun{j1},Km);
              
              % A op K
              [E, err] = evaluateOP(op,optype,A,K);
              [Em, err] = evaluateOP(op,optype,Am,Km);
              
              
              compareCPUGPU(Em,E);
              
              % K op A
              [E, err] = evaluateOP(op,optype,K,A);
              [Em, err] = evaluateOP(op,optype,Km,Am);
              
              
              compareCPUGPU(Em,E);
              
              % A.' op K
              [E, err] = evaluateOP(op,optype,A.',K);
              [Em, err] = evaluateOP(op,optype,Am.',Km);
              
              
              compareCPUGPU(Em,E);
              
              % K op A.'
              [E, err] = evaluateOP(op,optype,K,A.');
              [Em, err] = evaluateOP(op,optype,Km,Am.');
              
              
              compareCPUGPU(Em,E);
              
              % A' op K
              [E, err] = evaluateOP(op,optype,A',K);
              [Em, err] = evaluateOP(op,optype,Am',Km);
              
              compareCPUGPU(Em,E);
              
              % K op A'
              [E, err] = evaluateOP(op,optype,K,A');
              [Em, err] = evaluateOP(op,optype,Km,Am');
              
              
              compareCPUGPU(Em,E);
              
              
              % A.' op K.'
              [E, err] = evaluateOP(op,optype,A.',K.');
              [Em, err] = evaluateOP(op,optype,Am.',Km.');
              
              compareCPUGPU(Em,E);
              
              % K.' op A.'
              [E, err] = evaluateOP(op,optype,K.',A.');
              [Em, err] = evaluateOP(op,optype,Km.',Am.');
              
              
              compareCPUGPU(Em,E);
              
              % A' op K'
              [E, err] = evaluateOP(op,optype,A',K');
              [Em, err] = evaluateOP(op,optype,Am',Km');
              
              
              compareCPUGPU(Em,E);
              
              % K' op A'
              [E, err] = evaluateOP(op,optype,K',A');
              [Em, err] = evaluateOP(op,optype,Km',Am');
              
              
              compareCPUGPU(Em,E);
            end
          end
          
          
        end
        
      end
      
    end
  end
end
GPUtestLOG('***********************************************',0);

end

function [E ,err] = evaluateOP(op,optype,A,B)

global GPUtest

err = 0;

if (GPUtest.checkCompiler==1)
  switch optype
    case 1
      if (isa(A,'GPUtype'))
        E = comp_tmp(A);
      else
        E = eval([op '(A)']);
      end
    case 2
      if (isa(A,'GPUtype'))
        E = comp_tmp(A, B);
      else
        E = eval(['(A)' op '(B)']);
      end
    case 3
      if (isa(A,'GPUtype'))
        E = eval([op '(A)']);
        GPUzeros(E);
        comp_tmp(A, E);
      else
        E = eval([op '(A)']);
      end
    case 4
      if (isa(A,'GPUtype'))
        E = eval([op '(A, B)']);
        GPUzeros(E);
        comp_tmp(A, B, E);
      else
        E = eval([op '(A, B)']);
      end
      
  end
  
else
  
  switch optype
    case 1
      E = eval([op '(A)']);
    case 2
      E = eval(['(A)' op '(B)']);
    case 3
      if (isa(A,'GPUtype'))
        E = eval([op '(A)']);
        GPUzeros(E);
        eval(['GPU' op '(A, E)']);
      else
        E = eval([op '(A)']);
      end
    case 4
      if (isa(A,'GPUtype'))
        E = eval([op '(A, B)']);
        GPUzeros(E);
        eval(['GPU' op '(A, B, E)']);
      else
        E = eval([op '(A, B)']);
      end
      
  end
  
end
end
