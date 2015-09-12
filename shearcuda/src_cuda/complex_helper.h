/*
 * Author: Xavier Gibert-Serra (gibert@umiacs.umd.edu)
 *
 * Copyright 2012-2013 University of Maryland. All rights reserved.
 *
 */

#include "cuda_common.h"

// Basic operations on complex numbers
inline __device__ fComplex operator+(const fComplex& a, const fComplex& b)
{
    fComplex t = {a.x + b.x, a.y + b.y};
    return t;
}

inline __device__ dComplex operator+(const dComplex& a, const dComplex& b)
{
    dComplex t = {a.x + b.x, a.y + b.y};
    return t;
}

inline __device__ fComplex operator-(const fComplex& a, const fComplex& b)
{
    fComplex t = {a.x - b.x, a.y - b.y};
    return t;
}

inline __device__ dComplex operator-(const dComplex& a, const dComplex& b)
{
    dComplex t = {a.x - b.x, a.y - b.y};
    return t;
}

inline __device__ fComplex operator*(const fComplex& a, const fComplex& b)
{
    fComplex t = {a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y};
    return t;
}

inline __device__ dComplex operator*(const dComplex& a, const dComplex& b)
{
    dComplex t = {a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y};
    return t;
}

inline __device__ fComplex operator/(const fComplex& a, const fComplex& b)
{
    float d = b.x * b.x + b.y * b.y;
    fComplex t = {(a.x * b.x + a.y * b.y)/d, (a.y * b.x - a.x * b.y)/d};
    return t;
}

inline __device__ dComplex operator/(const dComplex& a, const dComplex& b)
{
    double d = b.x * b.x + b.y * b.y;
    dComplex t = {(a.x * b.x + a.y * b.y)/d, (a.y * b.x - a.x * b.y)/d};
    return t;
}

inline __device__ fComplex operator/(const fComplex& a, const float b)
{
    fComplex t = {a.x / b, a.y / b};
    return t;
}

inline __device__ dComplex operator/(const dComplex& a, const double b)
{
    dComplex t = {a.x / b, a.y / b};
    return t;
}

inline __device__ fComplex sqrt(const fComplex& a)
{
    float r = pow(a.x * a.x + a.y * a.y, 0.25f);
    float theta = atan2(a.y, a.x) / 2.f;
    fComplex t = {r*cos(theta), r*sin(theta)};
    return t;
}

inline __device__ dComplex sqrt(const dComplex& a)
{
    double r = pow(a.x * a.x + a.y * a.y, 0.25);
    double theta = atan2(a.y, a.x) / 2.0;
    dComplex t = {r*cos(theta), r*sin(theta)};
    return t;
}