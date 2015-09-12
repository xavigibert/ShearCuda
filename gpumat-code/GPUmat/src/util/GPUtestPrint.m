function GPUtestPrint
% GPUtestPrint Print GPUtest variable

%
%     Copyright (C) 2012  GP-you Group (http://gp-you.org)
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.


global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

%% Print configuration
disp(['* LOG file -> ' GPUtest.logFile]);

switch GPUtest.type
  case 1
    disp('* REAL/COMPLEX test ');
  case 2
    disp('* REAL test ');
  case 3
    disp('* COMPLEX test ');
    
end

disp('* Testing the following classes:');
for i=1:length(GPUtest.gpufun)
  disp(['   ' GPUtest.txtfun{i} ]);
  
end



disp(['* SINGLE tol -> ' num2str(GPUtest.tol.single)]);
disp(['* DOUBLE tol -> ' num2str(GPUtest.tol.double)]);

if (GPUtest.noCompareZeros==1)
  disp('* Error when comparing zeros');
else
  disp('* NO Error when comparing zeros');
end

if (GPUtest.stopOnError==1)
  disp('* STOP on error');
else
  disp('* NO stop on error');
end

if (GPUtest.fastMode==1)
  disp('* FAST MODE');
else
  disp('* NO fast mode');
end

if (GPUtest.memLeak==1)
  disp('* MEMORY LEAK CHECK');
else
  disp('* NO MEMORY LEAK CHECK');
end

if (GPUtest.checkPointers==1)
  disp('* POINTERS CHECK');
else
  disp('* NO POINTERS CHECK');
end

if (GPUtest.checkCompiler==1)
  disp('* COMPILER CHECK');
else
  disp('* NO COMPILER CHECK');
end

if (GPUtest.bigKernel==1)
  disp('* BIG KERNEL TEST');
else
  disp('* NO BIG KERNEL TEST');
end




