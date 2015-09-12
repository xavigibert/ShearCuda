classdef GPUtype < handle
  %GPUtype  Creates a GPU variable
  %   GPUtype is used to create a GPU variable on Matlab workspace.
  %   GPU calculations are performed by using operators or functions
  %   on GPUtype variables.
  %   Subclasses
  %   GPUsingle
  %   GPUdouble
  %
  %   Example
  %     GPUsingle(rand(100,100))
  %     Ah = rand(100);
  %     A  = GPUsingle(Ah);
  %     Bh = rand(100) + i*rand(100);
  %     B  = GPUsingle(Bh);
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

  properties
     
    slot = -1;
    
    
  end
   
  methods
    
    %% Constructor
    function p = GPUtype()
      % Nothing here     
      
    end
    
    
    %% DELETE
    
    function delete(p)
      % DELETE De-allocate a GPU variable from GPU memory.
      %        DELETE is equivalent to free in C language.
      %        De-allocate the variable from GPU memory.
      
      
      %--disp('delete GPUsingle');
      
      GPUtypeDelete(p.slot);
      
    end
    
    
  end
  
end
