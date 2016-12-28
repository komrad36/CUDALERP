///*******************************************************************
//*   main.cpp
//*   CUDALERP
//*
//*	Author: Kareem Omar
//*	kareem.omar@uah.edu
//*	https://github.com/komrad36
//*
//*	Last updated Dec 26, 2016
//*******************************************************************/
////
//// The file CUDALERP.h exposes two extremely high performance GPU
//// resize operations,
//// CUDALERP (bilinear interpolation), and 
//// CUDANERP (nearest neighbor interpolation).
////
//// CUDALERP offers superior accuracy to CUDA's built-in texture
//// interpolator at comparable performance. The accuracy if compiled
//// with -use-fast-math off is nearly equivalent to my CPU interpolator,
//// KLERP, while still being as fast as the built-in interpolation.
//// 
//// Particularly for large images, CUDALERP dramatically outperforms
//// even the highly tuned CPU AVX2 versions.
//// 
//// All functionality is contained in the header 'CUDALERP.h' and
//// the source file 'CUDALERP.cu' and has no external dependencies at all.
//// 
//// Note that these are intended for computer vision use(hence the speed)
//// and are designed for grayscale images.
//// 
//// The file 'main.cpp' is an example and speed test driver.
////
//
//#include <chrono>
//#include <cstring>
//#include <iostream>
//
//#include "CUDALERP.h"
//
//#define VC_EXTRALEAN
//#define WIN32_LEAN_AND_MEAN
//
//using namespace std::chrono;
//
//int main() {
//	constexpr auto warmups = 2000;
//	constexpr auto runs = 2000;
//
//	auto image = new uint8_t[4];
//	image[0] = 255;
//	image[1] = 255;
//	image[2] = 0;
//	image[3] = 0;
//
//	constexpr int oldw = 2;
//	constexpr int oldh = 2;
//	constexpr int neww = static_cast<int>(static_cast<double>(oldw) * 400.0);
//	constexpr int newh = static_cast<int>(static_cast<double>(oldh) * 1000.0);
//	const size_t total = static_cast<size_t>(neww)*static_cast<size_t>(newh);
//
//	// ------------- CUDALERP ------------
//
//	// setting cache and shared modes
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
//
//	// allocating and transferring image and binding to texture object
//	cudaChannelFormatDesc chandesc_img = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
//	cudaArray* d_img_arr;
//	cudaMallocArray(&d_img_arr, &chandesc_img, oldw, oldh, cudaArrayTextureGather);
//	cudaMemcpyToArray(d_img_arr, 0, 0, image, oldh * oldw, cudaMemcpyHostToDevice);
//	struct cudaResourceDesc resdesc_img;
//	memset(&resdesc_img, 0, sizeof(resdesc_img));
//	resdesc_img.resType = cudaResourceTypeArray;
//	resdesc_img.res.array.array = d_img_arr;
//	struct cudaTextureDesc texdesc_img;
//	memset(&texdesc_img, 0, sizeof(texdesc_img));
//	texdesc_img.addressMode[0] = cudaAddressModeClamp;
//	texdesc_img.addressMode[1] = cudaAddressModeClamp;
//	texdesc_img.readMode = cudaReadModeNormalizedFloat;
//	texdesc_img.filterMode = cudaFilterModePoint;
//	texdesc_img.normalizedCoords = 0;
//	cudaTextureObject_t d_img_tex = 0;
//	cudaCreateTextureObject(&d_img_tex, &resdesc_img, &texdesc_img, nullptr);
//
//	uint8_t* d_out = nullptr;
//	cudaMalloc(&d_out, total);
//
//	for (int i = 0; i < warmups; ++i) CUDALERP(d_img_tex, oldw, oldh, d_out, neww, newh);
//	auto start = high_resolution_clock::now();
//	for (int i = 0; i < runs; ++i) CUDALERP(d_img_tex, oldw, oldh, d_out, neww, newh);
//	auto end = high_resolution_clock::now();
//	auto sum = (end - start) / runs;
//
//	std::cout << "CUDALERP took " << static_cast<double>(sum.count()) * 1e-3 << " us." << std::endl;
//
//	std::cout << "Input stats: " << oldh << " rows, " << oldw << " cols." << std::endl;
//	std::cout << "Output stats: " << newh << " rows, " << neww << " cols." << std::endl;
//}
