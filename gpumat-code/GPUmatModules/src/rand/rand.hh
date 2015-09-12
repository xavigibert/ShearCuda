/*
     Copyright (C) 2012  GP-you Group (http://gp-you.org)
 
     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.
 
     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.
 
     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#if !defined(RAND_HH_)
#define RAND_HH_


/*************************************************
 * ERROR CODES
 *************************************************/
//
#define ERROR_CURAND_INIT  "Unable to initialize random number generator (ERROR code RAND.1)"
#define ERROR_CURAND_SEED  "Unable to set random number generator seed (ERROR code RAND.2)"

#define ERROR_RAND_COMPLEX  "Random number generator is not implemented for complex numbers (ERROR code RAND.3)"

#define ERROR_CURAND_GEN "Unable to generate random numbers (ERROR code RAND.4)"
#define ERROR_CURAND_DESTROY  "Unable to destroy random number generator (ERROR code RAND.5)"

#endif
