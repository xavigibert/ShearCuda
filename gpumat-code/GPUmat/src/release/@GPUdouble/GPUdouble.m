classdef GPUdouble < GPUtype
  %GPUdouble  Creates a GPU variable
  %   GPUdouble is used to create a GPU variable on Matlab workspace.
  %   GPU calculations are performed by using operators or functions
  %   on GPUdouble variables.
  %
  %   Example
  %     GPUdouble(rand(100,100))
  %     Ah = rand(100);
  %     A  = GPUdouble(Ah);
  %     Bh = rand(100) + i*rand(100);
  %     B  = GPUdouble(Bh);
  %
  
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
  
  
  methods
    
    %% Constructor
    function p = GPUdouble(varargin)
      p.slot = -1;
      switch nargin
        case 2
          p.slot = varargin{1};
        case 0
          p.slot = mxNumericArrayToGPUtypePtr(p, double([]));
        
        case 1
          A = varargin{1};
          if (isa(A,'GPUdouble'))
            p = clone(A);
            
          elseif (isa(A,'GPUsingle'))
            p.slot = GPUsingleToGPUdoublePtr(p, A);
            
          else
            p.slot = mxNumericArrayToGPUtypePtr(p, double(A));
          end
          
          
        otherwise
          error('Wrong number of input arguments');
      end
      
      
    end
    
    
    
    
  end
  
end
