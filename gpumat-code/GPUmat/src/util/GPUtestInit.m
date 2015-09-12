function GPUtestInit(varargin)
% GPUtestInit initializes GPUtest global variable
% GPUtest is the configuration variable used in tests.
%
% global variable GPUtest is used also in GPUtestLOG and compareCPUGPU
% functions
%
% Usage:
% GPUtestInit(option1,option2,...)
%
% Option is one of the following:
% single        - single precision only test
% double        - double precision only test
% single/double - both single and double precision tests
%
% real/complex  - both real and complex test
% real          - real test
% complex       - complex test
%
% stopOnError   - if you set this flag stops on error, otherwise
%                 write to log and continues the execution
%
% fastMode      - sets the fastMode flag.
%
%
% checkPointers - sets the checkPointers flag
%
% memLeak       - sets the memory leak check flag
%
% checkCompiler - checks the GPUmat compiler
% Example
% GPUtestInit 'real/complex' 'single' 'stopOnError'

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


% defaults
precision = 3;
type = 1;
stopOnError = 0;
fastMode = 0;
memLeak = 0;
checkPointers = 0;
checkCompiler = 0;
bigKernel = 0;

for i=1:length(varargin)
  % precision:
  % 1 - single
  % 2 - double
  % 3 - single/double
  %
  % real:
  % 1 - real/complex
  % 2 - real
  % 3 - complex
  
  
  switch varargin{i}
    case 'single'
      precision = 1;
    case 'double'
      precision = 2;
    case 'single/double'
      precision = 3;
    case 'real'
      type = 2;
    case 'complex'
      type = 3;
    case 'real/complex'
      type = 1;
    case 'stopOnError'
      stopOnError = 1;
    case 'fastMode'
      fastMode = 1;
    case 'memLeak'
      memLeak = 1;
    case 'checkPointers'
      checkPointers = 1;
    case 'checkCompiler'
      checkCompiler = 1;
    case 'bigKernel'
      bigKernel = 1;
      
    otherwise
      error('Unrecognized option');
  end
  
end

disp('* Init GPUtest');
global GPUtest

try
  type = GPUtest.type;
  disp('GPUtest is already initialized. User ''clear global'' to reset the variable.');
  return
catch
  % nothing
end

% output log
gmat = which('GPUstart');
gmat = gmat(1:end-11);
GPUtest.logFile = fullfile(gmat,'log.out');
fid=fopen(GPUtest.logFile,'w+');
fclose(fid);

% stop on error
GPUtest.stopOnError=stopOnError;

% display output also on screen. This flag is used by GPUtestLOG function
GPUtest.printDisplay = 1;

% GPUtest.N is used to set size of the matrix used in tests
GPUtest.N = [5 100 300];

% type
% 1 - real/complex
% 2 - real
% 3 - complex
GPUtest.type = type;

% precision:
% 1 - single
% 2 - double
% 3 - single/double

switch precision
  case 1
    GPUtest.gpufun = {@GPUsingle};
    GPUtest.cpufun = {@single};
    GPUtest.txtfun = {'single'};
    
  case 2
    GPUtest.gpufun = {@GPUdouble};
    GPUtest.cpufun = {@double};
    GPUtest.txtfun = {'double'};
    
  case 3
    GPUtest.gpufun = {@GPUsingle, @GPUdouble};
    GPUtest.cpufun = {@single, @double};
    GPUtest.txtfun = {'single', 'double'};
    
end

% Test function
% This function is used to generate the data in tests
GPUtest.testfun = @randn;

% Test tolerance
GPUtest.tol.single = 4e-6;
GPUtest.tol.double = 4e-15;

% Give error when comparing zeros
GPUtest.noCompareZeros = 0;

% Fast mode
GPUtest.fastMode = fastMode;

% Memory leak check
GPUtest.memLeak = memLeak;

% Check pointers
GPUtest.checkPointers = checkPointers;

% Check compiler
GPUtest.checkCompiler = checkCompiler;

% M and N are used in iterations  
if (fastMode)
  GPUtest.M = 5;
  GPUtest.N = 5;
else
  GPUtest.M = [5 100 300];
  GPUtest.N = [5 100 500];
end

% BIG Kernels flag
GPUtest.bigKernel = bigKernel;

GPUtestPrint;
