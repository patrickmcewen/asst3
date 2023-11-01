#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;

    int gridDim_x;
    int gridDim_y;
    int blockDim_x;
    int blockDim_y;

    int size_of_one_block;
    int size_of_one_row;
    int pow2Circles;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

GlobalConstants params;


// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"
#include "exclusiveScan.cu_inl"

#define BLOCKSIZE 256
#define SCAN_BLOCK_DIM   BLOCKSIZE  // needed by sharedMemExclusiveScan implementation

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif

static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr, bool check_pixel) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    if (check_pixel) {
        printf("old colors: %f, %f, %f\n", existingColor.x, existingColor.y, existingColor.z);
        printf("new colors: %f, %f, %f\n", newColor.x, newColor.y, newColor.z);
    }

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr, false);
            imgPtr++;
        }
    }
}

__global__ void kernelRenderPixels(int* circles_per_block_final, int* total_pairs) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("image width: %d, image height: %d\n", cuConstRendererParams.imageWidth, cuConstRendererParams.imageHeight);

    if (x >= cuConstRendererParams.imageWidth || y >= cuConstRendererParams.imageHeight) {
        return;
    }

    int total_pairs_offset = (blockIdx.y * cuConstRendererParams.gridDim_x) + blockIdx.x;
    int circles_per_block_offset = (cuConstRendererParams.size_of_one_row * blockIdx.y) + (cuConstRendererParams.size_of_one_block * blockIdx.x);
    int* circles_per_block_start = circles_per_block_final + circles_per_block_offset;
    int total_pairs_val = *(total_pairs + total_pairs_offset);
    bool check_pixel = false;
    if (x == 205 && y == 389) {
        printf("circle numbers: %d\n", total_pairs_val);
        check_pixel = true;
    }
    // dont launch kernel if num_circles_in_block = 0
    //printf("total pairs: %d", *total_pairs);
    //printf("x: %d, y: %d\n", x, y);
    for (int i = 0; i < total_pairs_val; i++) {
        int circle_ind = circles_per_block_start[i];
        //printf("circle_index: %d\n", circle_ind);
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[circle_ind*3]);

        // compute the bounding box of the circle. The bound is in integer
        // screen coordinates, so it's clamped to the edges of the screen.
        short imageWidth = cuConstRendererParams.imageWidth;
        short imageHeight = cuConstRendererParams.imageHeight;

        float invWidth = 1.f / imageWidth;
        float invHeight = 1.f / imageHeight;

        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (y * imageWidth + x)]);
        // for all pixels in the bonding box
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(x) + 0.5f),
                                            invHeight * (static_cast<float>(y) + 0.5f));
        shadePixel(i, pixelCenterNorm, p, imgPtr, check_pixel);
    }
}


// for each circle, loop over all blocks and check if the circle is contained inside
__global__ void kernelBoundCircles(int* circles_per_block) {
    int circle_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (circle_index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * circle_index;
    
    // read position and radius
    //printf("getting p and rad\n");
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[circle_index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    for (int x = 0; x < cuConstRendererParams.gridDim_x; x++) {
        for (int y = 0; y < cuConstRendererParams.gridDim_y; y++) {

            int circles_per_block_index = (cuConstRendererParams.size_of_one_row * y) + (cuConstRendererParams.size_of_one_block * x) + circle_index;

            float boxL = (float)(x * cuConstRendererParams.blockDim_x) / cuConstRendererParams.imageWidth;
            float boxR = (float)(x * (cuConstRendererParams.blockDim_x + 1)) / cuConstRendererParams.imageWidth;
            float boxT = (float)(y * cuConstRendererParams.blockDim_y) / cuConstRendererParams.imageHeight;
            float boxB = (float)(y * (cuConstRendererParams.blockDim_y + 1)) / cuConstRendererParams.imageHeight;
            //printf("top: %f, bottom: %f, left: %f, right: %f, p.x: %f, p.y: %f, rad: %f\n", boxT, boxB, boxL, boxR, p.x, p.y, rad);
            //printf("accessing %d index vs size of circles_per_block: %d\n", circles_per_block_index, cuConstRendererParams.numCircles * cuConstRendererParams.gridDim_x * cuConstRendererParams.gridDim_y);
            //printf("image width: %d, image height: %d\n", cuConstRendererParams.imageWidth, cuConstRendererParams.imageHeight);
            circles_per_block[circles_per_block_index] = circleInBox(p.x, p.y, rad, boxL, boxR, boxT, boxB);
            //printf("result was %d\n", circles_per_block[circles_per_block_index]);
        }
    }
}

__global__ void kernelExclusiveScan(int* circles_per_block_start, int x, int y, uint* prefixSumScratch) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= cuConstRendererParams.pow2Circles)
        return;
    
    //int old_data = circles_per_block_start[index];
    
    circles_per_block_start[index] = warpScanExclusive(index, circles_per_block_start[index], prefixSumScratch, cuConstRendererParams.pow2Circles);
    //if (circles_per_block_start[index])
        //printf("warp scan result for index %d is %d\n", index, circles_per_block_start[index]);

    //printf("for index %d, input data %d, output data %d\n", index, old_data, circles_per_block_start[index]);
}

__global__ void get_repeats_final(int* input, int* output, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length - 1 && (input[index] < input[index+1])) {
        output[input[index]] = index;
        //printf("marked a repeat\n");
    }
}

__global__ void get_total_pairs(int* input, int length, int* total_pairs) {
    total_pairs[0] = input[length-1];
    //if (total_pairs[0] > 0)
        //printf("total_pairs: %d\n", total_pairs[0]);
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    params.blockDim_x = 16;
    params.blockDim_y = 16;

    params.gridDim_x = (params.imageWidth + params.blockDim_x - 1) / params.blockDim_x;
    params.gridDim_y =  (params.imageHeight + params.blockDim_y - 1) / params.blockDim_y;

    params.pow2Circles = nextPow2(params.numCircles);
    params.size_of_one_row = params.pow2Circles * params.gridDim_x;
    params.size_of_one_block = params.pow2Circles;
    

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    //incorrect implementation
    // 256 threads per block is a healthy number
    /*dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();*/

    dim3 blockDimCircles(256, 1);
    dim3 gridDimCircles((params.numCircles + blockDimCircles.x - 1) / blockDimCircles.x);
    int* circles_per_block = nullptr; // flattened 2d array
    int* circles_per_block_final = nullptr;
    cudaMalloc(&circles_per_block, sizeof(int) * params.pow2Circles * params.gridDim_x * params.gridDim_y);
    cudaMalloc(&circles_per_block_final, sizeof(int) * params.pow2Circles * params.gridDim_x * params.gridDim_y);

    kernelBoundCircles<<<gridDimCircles, blockDimCircles>>>(circles_per_block);


    cudaCheckError(cudaDeviceSynchronize());

    //launch exclusive scans for each block
    for (int x = 0; x < params.gridDim_x; x++) {
        for (int y = 0; y < params.gridDim_y; y++) {
            //printf("x: %d, y: %d\n", x, y);
            dim3 blockDimScan(256, 1);
            dim3 gridDimScan((params.pow2Circles + blockDimScan.x - 1) / blockDimScan.x);
            uint* prefixSumScratch = nullptr;
            cudaMalloc(&prefixSumScratch, sizeof(uint) * params.pow2Circles * 2);
            int circles_per_block_offset = (params.size_of_one_row * y) + (params.size_of_one_block * x);
            int* circles_per_block_start = circles_per_block + circles_per_block_offset;
            kernelExclusiveScan<<<gridDimScan, blockDimScan>>>(circles_per_block_start, x, y, prefixSumScratch);
            cudaDeviceSynchronize();
        }
        //printf("\n");
    }


    cudaCheckError(cudaDeviceSynchronize());

    int* print_data = (int*)malloc(sizeof(int) * params.pow2Circles * params.gridDim_x * params.gridDim_y);
    cudaMemcpy(print_data, circles_per_block, sizeof(int) * params.pow2Circles * params.gridDim_x * params.gridDim_y, cudaMemcpyDeviceToHost);

    int* total_pairs = nullptr;
    cudaMalloc(&total_pairs, sizeof(int) * params.gridDim_x * params.gridDim_y);

    for (int x = 0; x < params.gridDim_x; x++) {
        for (int y = 0; y < params.gridDim_y; y++) {
            int circles_per_block_offset = (params.size_of_one_row * y) + (params.size_of_one_block * x);
            int* circles_per_block_start = circles_per_block + circles_per_block_offset;
            get_total_pairs<<<1, 1>>>(circles_per_block_start, params.pow2Circles, total_pairs + (y * params.gridDim_x) + x);
        }
    }

    cudaCheckError(cudaDeviceSynchronize());

    for (int x = 0; x < params.gridDim_x; x++) {
        for (int y = 0; y < params.gridDim_y; y++) {
            int circles_per_block_offset = (params.size_of_one_row * y) + (params.size_of_one_block * x);
            int* circles_per_block_start = circles_per_block + circles_per_block_offset;
            int* circles_per_block_final_start = circles_per_block_final + circles_per_block_offset;
            get_repeats_final<<<gridDimCircles, blockDimCircles>>>(circles_per_block_start, circles_per_block_final_start, params.pow2Circles);
        }
    }
    
    /*int* total_pairs_print = (int*)malloc(sizeof(int) * params.gridDim_x * params.gridDim_y);
    cudaMemcpy(total_pairs_print, total_pairs, sizeof(int) * params.gridDim_x * params.gridDim_y, cudaMemcpyDeviceToHost);
    int* print_data2 = (int*)malloc(sizeof(int) * params.pow2Circles * params.gridDim_x * params.gridDim_y);
    cudaMemcpy(print_data2, circles_per_block_final, sizeof(int) * params.pow2Circles * params.gridDim_x * params.gridDim_y, cudaMemcpyDeviceToHost);
    printf("copied data\n");
    for (int x = 0; x < params.gridDim_x; x++) {
        for (int y = 0; y < params.gridDim_y; y++) {
            int total_pairs_offset = (y * params.gridDim_x) + x;
            int circles_per_block_offset = (params.size_of_one_row * y) + (params.size_of_one_block * x);
            int* circles_per_block_start = print_data2 + circles_per_block_offset;
            int* old_data_start = print_data + circles_per_block_offset;
            printf("index %d and %d: (total pairs %d)", x, y, *(total_pairs_print + total_pairs_offset));
            for (int j = 0; j < params.pow2Circles; j++) {
                printf("%d ", circles_per_block_start[j]);
            }
            printf("\n");
            printf("input data for that one: ");
            for (int j = 0; j < params.pow2Circles; j++) {
                printf("%d ", old_data_start[j]);
            }
            printf("\n");
        }
        if (x > params.gridDim_x / 2) break;
    }

    cudaCheckError(cudaDeviceSynchronize());*/

    // pixel parallel only
    dim3 blockDim(params.blockDim_x, params.blockDim_y);
    dim3 gridDim(params.gridDim_x, params.gridDim_y);
    //printf("blockDims: %d %d, gridDims: %d %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    //printf("imageWidth: %d, height: %d\n", params.imageWidth, params.imageHeight);
    //printf("grid dims are x- %d and y- %d\n", gridDim.x, gridDim.y);
    // go one thread per block instead of 1 thread per pixel
    kernelRenderPixels<<<gridDim, blockDim>>>(circles_per_block_final, total_pairs);

    float* image_data_print = (float*)malloc(sizeof(float) * params.imageWidth * params.imageHeight * 4);
    cudaMemcpy(image_data_print, cudaDeviceImageData, sizeof(float) * params.imageWidth * params.imageHeight * 4, cudaMemcpyDeviceToHost);
    float4* data = (float4*)(&image_data_print[4 * (389 * params.imageWidth + 205)]);
    printf("205, 389: %f, %f, %f\n", data->x, data->y, data->z);
    cudaDeviceSynchronize();
}

