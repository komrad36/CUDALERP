/*******************************************************************
*   CUDALERP.h
*   CUDALERP
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Jan 7, 2016
*******************************************************************/
//
// The file CUDALERP.h exposes two extremely high performance GPU
// resize operations,
// CUDALERP (bilinear interpolation), and 
// CUDANERP (nearest neighbor interpolation), for 8-bit unsigned
// integer (i.e. grayscale) data.
//
// For 32-bit float data, see the CUDAFLERP project instead.
//
// CUDALERP offers superior accuracy to CUDA's built-in texture
// interpolator at comparable performance. The accuracy if compiled
// with -use-fast-math off is nearly equivalent to my CPU interpolator,
// KLERP, while still being as fast as the built-in interpolation.
// 
// Particularly for large images, CUDALERP dramatically outperforms
// even the highly tuned CPU AVX2 versions.
// 
// All functionality is contained in the header 'CUDALERP.h' and
// the source file 'CUDALERP.cu' and has no external dependencies at all.
// 
// Note that these are intended for computer vision use(hence the speed)
// and are designed for grayscale images.
// 
// The file 'main.cpp' is an example and speed test driver.
//

#pragma once

#include "cuda_runtime.h"

#include <cstdint>

#ifdef __INTELLISENSE__
#include <algorithm>
#define asm(x)
#include "device_launch_parameters.h"
#define __CUDACC__
#include "device_functions.h"
#undef __CUDACC__
#endif

void CUDALERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, uint8_t* __restrict const d_out, const uint32_t neww, const uint32_t newh);

void CUDANERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, uint8_t* __restrict const d_out, const uint32_t neww, const uint32_t newh);
