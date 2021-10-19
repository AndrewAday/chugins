#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <vector>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "FFTConvolver.h"
#include "Utilities.h"


template<typename T>
void SimpleConvolve(const T* input, size_t inLen, const T* ir, size_t irLen, T* output)
{
  if (irLen > inLen)
  {
    SimpleConvolve(ir, irLen, input, inLen, output);
    return;
  }

  ::memset(output, 0, (inLen+irLen-1) * sizeof(T));


  // optimized time-domain conv

  for (size_t n=0; n<irLen; ++n)  // as input is entering
  {
    for (size_t m=0; m<=n; ++m)
    {
      output[n] += ir[m] * input[n-m];
    }
  }

  for (size_t n=irLen; n<inLen; ++n) // as input is overlapped
  {
    for (size_t m=0; m<irLen; ++m)
    {
      output[n] += ir[m] * input[n-m];
    }
  }

  for (size_t n=inLen; n<inLen+irLen-1; ++n) // as input is moving away
  {
    for (size_t m=n-inLen+1; m<irLen; ++m)
    {
      output[n] += ir[m] * input[n-m];
    }
  }
}


static bool TestConvolver(size_t inputSize,
                          size_t irSize,
                          size_t blockSizeMin,
                          size_t blockSizeMax,
                          size_t blockSizeConvolver,
                          bool refCheck)
{
  // Prepare input and IR
  std::vector<fftconvolver::Sample> in(inputSize);  // sample is float, def in utilities.h
  for (size_t i=0; i<inputSize; ++i)
  {
    in[i] = 0.1f * static_cast<fftconvolver::Sample>(i+1);
    // in = [1, 2, 3, ...]
  }

  std::vector<fftconvolver::Sample> ir(irSize);
  for (size_t i=0; i<irSize; ++i)
  {
    ir[i] = 0.1f * static_cast<fftconvolver::Sample>(i+1);
    // ir = [1, 2, 3, ...]
  }

  // Simple convolver, brute force
  if (refCheck)  // check with ground truth time-domain conv
  {
    std::vector<fftconvolver::Sample> outSimple(in.size() + ir.size() - 1, fftconvolver::Sample(0.0));  // init all values to 0.0
    SimpleConvolve(&in[0], in.size(), &ir[0], ir.size(), &outSimple[0]);
  }

  // TODO:  finish going over this example, figure out how to link this to ConvRev Chugin
  // FFT convolver

  // output buffer post-convolution, prepopulate with 0s
  std::vector<fftconvolver::Sample> out(in.size() + ir.size() - 1, fftconvolver::Sample(0.0));
  {
    fftconvolver::FFTConvolver convolver;
    convolver.init(blockSizeConvolver, &ir[0], ir.size());
    std::vector<fftconvolver::Sample> inBuf(blockSizeMax);
    size_t processedOut = 0;  // how much output convolution have we returned
    size_t processedIn = 0;  // how much of input signal have we convolved
    while (processedOut < out.size())  // the entire convolution
    {
      // rand blocksize from blocksizemin to blocksizemax
      const size_t blockSize = blockSizeMin + (static_cast<size_t>(rand()) % (1+(blockSizeMax-blockSizeMin)));


      const size_t remainingOut = out.size() - processedOut;
      const size_t remainingIn = in.size() - processedIn;

      const size_t processingOut = std::min(remainingOut, blockSize);
      const size_t processingIn = std::min(remainingIn, blockSize);

      memset(&inBuf[0], 0, inBuf.size() * sizeof(fftconvolver::Sample));  // zero the in buffer
      if (processingIn > 0)
      {
        // could probably avoid this memcpy if we pass processedIn param to fftconvolver.process
        memcpy(&inBuf[0], &in[processedIn], processingIn * sizeof(fftconvolver::Sample));
      } // buffer blocksize amount of input samples

      // after all input samples are processed, we are only a stream of 0s
      // as convolver finishes convolving the earlier segments
      convolver.process(&inBuf[0], &out[processedOut], processingOut /*how many samples to write to out buffer */);
      // write convolution result to out buffer

      processedOut += processingOut;
      processedIn += processingIn;
    }
  }

  if (refCheck)
  {
    size_t diffSamples = 0;
    const double absTolerance = 0.001 * static_cast<double>(ir.size());
    const double relTolerance = 0.0001 * ::log(static_cast<double>(ir.size()));
    for (size_t i=0; i<outSimple.size(); ++i)
    {
      const double a = static_cast<double>(out[i]);
      const double b = static_cast<double>(outSimple[i]);
      if (::fabs(a) > 1.0 && ::fabs(b) > 1.0)
      {
        const double absError = ::fabs(a-b);
        const double relError = absError / b;
        if (relError > relTolerance && absError > absTolerance)
        {
          ++diffSamples;
        }
      }
    }
    printf("Correctness Test (input %d, IR %d, blocksize %d-%d) => %s\n", static_cast<int>(inputSize), static_cast<int>(irSize), static_cast<int>(blockSizeMin), static_cast<int>(blockSizeMax), (diffSamples == 0) ? "[OK]" : "[FAILED]");
    return (diffSamples == 0);
  }
  else
  {
    printf("Performance Test (input %d, IR %d, blocksize %d-%d) => Completed\n", static_cast<int>(inputSize), static_cast<int>(irSize), static_cast<int>(blockSizeMin), static_cast<int>(blockSizeMax));
    return true;
  }
}


#define TEST_PERFORMANCE
#define TEST_CORRECTNESS
#define TEST_FFTCONVOLVER

int main()
{
#if defined(TEST_CORRECTNESS) && defined(TEST_FFTCONVOLVER)
  fputs(fftconvolver::SSEEnabled() ? "SSE ENabled true\n" : "SSE ENabled false\n", stdout);
  /*
  - in size
  - ir size
  - block min
  - block max
  - conv block size
  */
  TestConvolver(1, 1, 1, 1, 1, true);
  TestConvolver(2, 2, 2, 2, 2, true);
  TestConvolver(3, 3, 3, 3, 3, true);

  TestConvolver(3, 2, 2, 2, 2, true);
  TestConvolver(4, 2, 2, 2, 2, true);
  TestConvolver(4, 3, 2, 2, 2, true);
  TestConvolver(9, 4, 3, 3, 2, true);
  TestConvolver(171, 7, 5, 5, 5, true);
  TestConvolver(1979, 17, 7, 7, 5, true);
  TestConvolver(100, 10, 3, 5, 5, true);
  TestConvolver(123, 45, 12, 34, 34, true);

  TestConvolver(2, 3, 2, 2, 2, true);
  TestConvolver(2, 4, 2, 2, 2, true);
  TestConvolver(3, 4, 2, 2, 2, true);
  TestConvolver(4, 9, 3, 3, 3, true);
  TestConvolver(7, 171, 5, 5, 5, true);
  TestConvolver(17, 1979, 7, 7, 7, true);
  TestConvolver(10, 100, 3, 5, 5, true);
  TestConvolver(45, 123, 12, 34, 34, true);

  TestConvolver(100000, 1234, 100,  128,  128, true);
  TestConvolver(100000, 1234, 100,  256,  256, true);
  TestConvolver(100000, 1234, 100,  512,  512, true);
  TestConvolver(100000, 1234, 100, 1024, 1024, true);
  TestConvolver(100000, 1234, 100, 2048, 2048, true);

  TestConvolver(100000, 4321, 100,  128,  128, true);
  TestConvolver(100000, 4321, 100,  256,  256, true);
  TestConvolver(100000, 4321, 100,  512,  512, true);
  TestConvolver(100000, 4321, 100, 1024, 1024, true);
  TestConvolver(100000, 4321, 100, 2048, 2048, true);
#endif


using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#if defined(TEST_PERFORMANCE) && defined(TEST_FFTCONVOLVER)
  // TODO: time many powers of 2 to find optimal blocksize
  auto t1 = high_resolution_clock::now();
  TestConvolver(3*60*44100, 20*44100, 50, 100, 1024, false);
  auto t2 = high_resolution_clock::now();

  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;

  printf("%f ms \n", ms_double.count());
#endif

  return 0;
}
