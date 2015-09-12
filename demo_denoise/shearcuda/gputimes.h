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

#ifdef _WIN32
    // includes, system
    #define WINDOWS_LEAN_AND_MEAN
    #include <windows.h>
    #undef min
    #undef max
#else
    // includes, system
    #include <ctime>
    #include <sys/time.h>
#endif

#include "shearcuda_global.h"

// Class containing timers
class SHEARCUDASHARED_EXPORT StopWatch
{
public:
    StopWatch() :
        m_numCalls(0),
        m_totalTime(0.f)
    {}

    void reset()
    {
        m_numCalls = 0;
        m_totalTime = 0.f;
    }

    void start()
    {
#ifdef _WIN32
        QueryPerformanceCounter((LARGE_INTEGER*) &m_startTime);
#else
        gettimeofday( &m_startTime, 0);
#endif
    }

    void stop()
    {
        float diff_time;
#ifdef _WIN32
        LARGE_INTEGER end_time;
        QueryPerformanceCounter((LARGE_INTEGER*) &end_time);
        diff_time = (float)
            (((double) end_time.QuadPart - (double) m_startTime.QuadPart) / s_freq);
#else
        struct timeval t_time;
        gettimeofday( &t_time, 0);

        // time difference in milli-seconds
        diff_time = (float) (1000.0 * ( t_time.tv_sec - m_startTime.tv_sec) 
                            + (0.001 * (t_time.tv_usec - m_startTime.tv_usec)) );
#endif
        m_totalTime += diff_time;
        m_numCalls++;

    }

    int m_numCalls;
    float m_totalTime;

#ifdef _WIN32
    LARGE_INTEGER m_startTime;
    //! tick frequency
    static double s_freq;
#else
    struct timeval m_startTime;
#endif
};

// This class provides a common object to keep track of timing for each
// kernel call
class SHEARCUDASHARED_EXPORT GpuTimes
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
        reduceNormErr256,
        reduceSum256,
        zeroPad,
        prepareMyerFilters,
        convert,
        cudaMemcpy,
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

private:
    StopWatch m_timer[numTimers];
    bool m_bEnabled;
};

#endif  // GPUTIMES_H
