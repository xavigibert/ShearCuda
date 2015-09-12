function [t] = type_of_shear_dict( shear )
% This function returns the type of data this shearlet dictionary can
% operate on. Possible return values are:
%   'single', 'double', 'GPUsingle', or 'GPUdouble'

    if iscell( shear )
        t = class( shear{1} );
    else
        t = class( shear.filter{1} );
    end
    
end