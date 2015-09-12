/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright (C) 2013 University of Maryland. All rights reserved.
 *
 */

#include "CracksGpuTimes.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include <string.h>
#include <mex.h>
#include <algorithm>
#include <vector>
#include "GPUmat.hh"

static char s_funcNames[][32] = {"llCrackFeatures", "dotProductsFeatures",
                                 "graphAffinitiesFromDotProd", "graphTerminals", "graphCut8"};

CracksGpuTimes::CracksGpuTimes()
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

void CracksGpuTimes::resetTimers()
{
    for( int i = 0; i < numTimers; i++ )
        m_timer[i].reset();
}

void CracksGpuTimes::startTimer(TimerID id)
{
    if( m_bEnabled )
    {
        cudaDeviceSynchronize();
        m_timer[id].start();
    }
}

void CracksGpuTimes::stopTimer(TimerID id)
{
    if( m_bEnabled )
    {
        cudaDeviceSynchronize();
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
static bool operator<(const TimeIdx& a, const TimeIdx& b) { return a.time > b.time; }
static bool timeNotLessThan(TimeIdx a, TimeIdx b) { return a.time > b.time; }

void CracksGpuTimes::displayTimers()
{
    float totalTime = 0;
    for( int i = 0; i < numTimers; i++ )
        totalTime += m_timer[i].m_totalTime;

    if( totalTime == 0 )
        return;

    for( int i = 0; i < 61; i++ ) mexPrintf("-");
    mexPrintf("\n|                     Step | cnt |  total  |  avg   |  rel  |\n");
    for( int i = 0; i < 61; i++ ) mexPrintf("-");
    mexPrintf("\n");

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
            mexPrintf("| %24s | %3d | %7.3f | %6.3f | %4.1f%% |\n", s_funcNames[sorted[i].timerIdx],
                      t.m_numCalls, t.m_totalTime,
                      t.m_totalTime/t.m_numCalls,
                      100.0 * t.m_totalTime/totalTime);
    }

    for( int i = 0; i < 61; i++ ) mexPrintf("-");
    mexPrintf("\nTOTAL TIME: %.3f msec \n", totalTime);
}

CracksGpuTimes* CracksGpuTimes::getGpuTimesObject()
{
    mxArray* prhs[1];
    prhs[0] = mxCreateDoubleScalar( 2 );
    mxArray* plhs[1];
    mexCallMATLAB(1, plhs, 1, prhs, "cracks_timers");

    return *((CracksGpuTimes**)mxGetData(plhs[0]));
}
