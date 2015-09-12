/*
     Copyright (C) 2013  University of Maryland
 
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

#ifndef GPUTIMES_H
#define GPUTIMES_H

#include "StopWatch.h"

// This class provides a common object to keep track of timing for each
// kernel call
class GpuTimes
{
public:
    // IDs for each possible kernel or function call
    enum TimerID
    {
        fftFwd = 0,
        fftInv,
        nppsMax,
        nppsMin,
        modulateAndNormalize,
        addVector,
        sumVectors,
        mulMatrixByScalar,
        atrousSubsample,
        atrousUpsample,
        atrousConvolution,
        hardThreshold,
        softThreshold,
        symExt,
        scalarVectorMul,
        mrdwtRow,
        mrdwtCol,
        mirdwtRow,
        mirdwtCol,
        realToComplex,
        complexToReal,
        reduceMaxAbsVal256,
        reduceNorm256,
        reduceSum256,
        zeroPad,
        testTimer,
        testTimer1,
        testTimer2,
        testTimer3,
        testTimer4,
        testTimer5,
        numTimers
    };

    GpuTimes();

    void resetTimers();
    void startTimer(TimerID id);
    void stopTimer(TimerID id);
    void displayTimers();
    void enable() { m_bEnabled = true; }
    void disable() { m_bEnabled = false; }
    static GpuTimes* getGpuTimesObject();

private:
    StopWatch m_timer[numTimers];
    bool m_bEnabled;
};

#endif  // GPUTIMES_H
