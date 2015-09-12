/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright (C) 2013 University of Maryland. All rights reserved.
 *
 */

#ifndef CRACKSGPUTIMES_H
#define CRACKSGPUTIMES_H

#include "StopWatch.h"

// This class provides a common object to keep track of timing for each
// kernel call
class CracksGpuTimes
{
public:
    // IDs for each possible kernel or function call
    enum TimerID
    {
        llCrackFeatures = 0,
        dotProductsFeatures = 1,
        graphAffinitiesFromDotProd = 2,
        graphTerminals = 3,
        graphCut8 = 4,
        numTimers
    };

    CracksGpuTimes();

    void resetTimers();
    void startTimer(TimerID id);
    void stopTimer(TimerID id);
    void displayTimers();
    void enable() { m_bEnabled = true; }
    void disable() { m_bEnabled = false; }
    static CracksGpuTimes* getGpuTimesObject();

private:
    StopWatch m_timer[numTimers];
    bool m_bEnabled;
};

#endif  // CRACKSGPUTIMES_H
