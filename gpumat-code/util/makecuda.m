function makecuda(base, include, clean)

% default
if (~exist('clean','var'))
  clean = 0;
end

% load nvidia settings (CUDA_ROOT, CC_BIN)
cuda = nvidiasettings;

%%disp('**** MAKE CUDA ****');

%% Build kernel mykernel.cubin
%%disp('Building CUDA Kernel');
%%disp('* Calling nvcc');

%nvidiacmd = ['nvcc -arch sm_10 -maxrregcount=32 ' include ' -m32 -cubin -o "numerics.cubin" "numerics.cu"'];

%base = 'numerics';
arch = cuda.arch;
for i=1:length(arch)
  outputfile = ['".' filesep base arch{i} '.cubin"'];
  inputfile  = ['".' filesep base '.cu"'];
  if (clean==1)
    filename = [base arch{i} '.cubin'];
    if (exist(filename,'file'))
      cmd = ['delete ' filename ];
      disp(cmd);
      eval(cmd);
    end
  else
    
    
    clinclude = '';
    switch computer
      case {'PCWIN64'}
        machine = '-m64';
        clinclude = locateCL;
      case {'PCWIN'}
        machine = '-m32';
      case {'GLNXA64'}
        machine = '-m64';
      otherwise
        machine = '-m32';
    end
    
    infile = java.io.File([base '.cu']);
    outfile = java.io.File([base arch{i} '.cubin']);
    
    if (infile.lastModified > outfile.lastModified)
      nvidiacmd = ['nvcc -arch sm_' arch{i} ' ' clinclude ' -maxrregcount=32 ' include ' ' machine ' -cubin -o  ' outputfile ' ' inputfile];
      
      disp(nvidiacmd);
      system(nvidiacmd);
    else
      disp([inputfile '-> nothing to be done']);
    end
    
    
    
    
    
  end
end


%disp(nvidiacmd);
%system(nvidiacmd);

end







