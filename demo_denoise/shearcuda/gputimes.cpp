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

#include "gputimes.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "cutil_inline_runtime.h"

static char s_funcNames[][32] = {"fftFwd", "fftInv", "nppsMax", "nppsMin",
    "modulateAndNormalize", "addVector", "sumVectors", "mulMatrixByScalar",
    "atrousSubsample", "atrousUpsample", "atrousConvolution", "hardThreshold",
    "softThreshold", "symExt", "scalarVectorMul", "mrdwtRow", "mrdwtCol",
    "mirdwtRow", "mirdwtCol", "realToComplex", "complexToReal",
    "reduceMaxAbsVal256", "reduceNorm256", "reduceNormErr256", "reduceSum256", "zeroPad",
    "prepareMyerFilters", "convert", "cudaMemcpy", "testTimer",
    "testTimer1", "testTimer2", "testTimer3", "testTimer4", "testTimer5"
};

#ifdef _WIN32
double StopWatch::s_freq = 0.0;
#endif

GpuTimes::GpuTimes()
{
    m_bEnabled = true;

#ifdef _WIN32
    // helper variable
    LARGE_INTEGER temp;

    // get the tick frequency from the OS
    QueryPerformanceFrequency((LARGE_INTEGER*) &temp);

    // convert to type in which it is needed
    StopWatch::s_freq = ((double) temp.QuadPart) / 1000.0;
#endif
}

void GpuTimes::resetTimers()
{
    for( int i = 0; i < numTimers; i++ )
        m_timer[i].reset();
}

void GpuTimes::startTimer(TimerID id)
{
    if( m_bEnabled )
    {
        cutilDeviceSynchronize();
        m_timer[id].start();
    }
}

void GpuTimes::stopTimer(TimerID id)
{
    if( m_bEnabled )
    {
        cutilDeviceSynchronize();
        m_timer[id].stop();
    }
}

class TimeIdx
{
public:
    TimeIdx(): timerIdx(0), time(0.0) {}
    TimeIdx(const TimeIdx& src)
    {
        timerIdx = src.timerIdx;
        time = src.time;
    }

    int timerIdx;
    double time;
};
bool operator<(const TimeIdx& a, const TimeIdx& b) { return a.time > b.time; }
bool timeNotLessThan(TimeIdx a, TimeIdx b) { return a.time > b.time; }

void GpuTimes::displayTimers()
{
    float totalTime = 0;
    for( int i = 0; i < testTimer; i++ )
        totalTime += m_timer[i].m_totalTime;

    if( totalTime == 0 )
        return;

    for( int i = 0; i < 61; i++ ) printf("-");
    printf("\n|                     Step | cnt |  total  |  avg   |  rel  |\n");
    for( int i = 0; i < 61; i++ ) printf("-");
    printf("\n");

    // Sort times
    std::vector<TimeIdx> sorted( numTimers );
    for( int i = 0; i < numTimers; i++ ) {
        sorted[i].timerIdx = i;
        sorted[i].time = m_timer[i].m_totalTime;
    }
    std::stable_sort( sorted.begin(), sorted.end(), timeNotLessThan);

    for( int i = 0; i < numTimers; i++ )
    {
        const StopWatch& t = m_timer[sorted[i].timerIdx];
        if( t.m_numCalls > 0 )
            printf("| %24s | %3d | %7.3f | %6.3f | %4.1f%% |\n", s_funcNames[sorted[i].timerIdx],
                      t.m_numCalls, t.m_totalTime,
                      t.m_totalTime/t.m_numCalls,
                      100.0 * t.m_totalTime/totalTime);
    }

    for( int i = 0; i < 61; i++ ) printf("-");
    printf("\nTOTAL TIME: %.3f msec \n", totalTime);
}

