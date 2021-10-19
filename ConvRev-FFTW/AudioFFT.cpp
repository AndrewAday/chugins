// ==================================================================================
// Copyright (c) 2017 HiFi-LoFi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ==================================================================================

#include "AudioFFT.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#if defined (AUDIOFFT_FFTW3)
  #define AUDIOFFT_FFTW3_USED
  #include <fftw3.h>
#else
  #if !defined(AUDIOFFT_OOURA)
    #define AUDIOFFT_OOURA
  #endif
  #define AUDIOFFT_OOURA_USED
  #include <vector>
#endif


namespace audiofft
{

  namespace detail
  {

    class AudioFFTImpl
    {
    public:
      AudioFFTImpl() = default;
      AudioFFTImpl(const AudioFFTImpl&) = delete;
      AudioFFTImpl& operator=(const AudioFFTImpl&) = delete;
      virtual ~AudioFFTImpl() = default;
      virtual void init(size_t size) = 0;
      virtual void fft(const float* data, float* re, float* im) = 0;
      virtual void ifft(float* data, const float* re, const float* im) = 0;
    };


    constexpr bool IsPowerOf2(size_t val)
    {
      return (val == 1 || (val & (val-1)) == 0);
    }


    template<typename TypeDest, typename TypeSrc>
    void ConvertBuffer(TypeDest* dest, const TypeSrc* src, size_t len)
    {
      for (size_t i=0; i<len; ++i)
      {
        dest[i] = static_cast<TypeDest>(src[i]);
      }
    }


    template<typename TypeDest, typename TypeSrc, typename TypeFactor>
    void ScaleBuffer(TypeDest* dest, const TypeSrc* src, const TypeFactor factor, size_t len)
    {
      for (size_t i=0; i<len; ++i)
      {
        dest[i] = static_cast<TypeDest>(static_cast<TypeFactor>(src[i]) * factor);
      }
    }

  } // End of namespace detail


  // ================================================================


#ifdef AUDIOFFT_FFTW3_USED


  /**
   * @internal
   * @class FFTW3FFT
   * @brief FFT implementation using FFTW3 internally (see fftw.org)
   */
  class FFTW3FFT : public detail::AudioFFTImpl
  {
  public:
    FFTW3FFT() :
      detail::AudioFFTImpl(),
      _size(0),
      _complexSize(0),
      _planForward(0),
      _planBackward(0),
      _data(0),
      _re(0),
      _im(0)
    {
    }

    FFTW3FFT(const FFTW3FFT&) = delete;
    FFTW3FFT& operator=(const FFTW3FFT&) = delete;

    virtual ~FFTW3FFT()
    {
      init(0);
    }

    virtual void init(size_t size) override
    {
      if (_size != size)
      {
        if (_size > 0)
        {
          fftwf_destroy_plan(_planForward);
          fftwf_destroy_plan(_planBackward);
          _planForward = 0;
          _planBackward = 0;
          _size = 0;
          _complexSize = 0;

          if (_data)
          {
            fftwf_free(_data);
            _data = 0;
          }

          if (_re)
          {
            fftwf_free(_re);
            _re = 0;
          }

          if (_im)
          {
            fftwf_free(_im);
            _im = 0;
          }
        }

        if (size > 0)
        {
          _size = size;
          _complexSize = AudioFFT::ComplexSize(_size);
          const size_t complexSize = AudioFFT::ComplexSize(_size);
          _data = reinterpret_cast<float*>(fftwf_malloc(_size * sizeof(float)));
          _re = reinterpret_cast<float*>(fftwf_malloc(complexSize * sizeof(float)));
          _im = reinterpret_cast<float*>(fftwf_malloc(complexSize * sizeof(float)));

          fftw_iodim dim;
          dim.n = static_cast<int>(size);
          dim.is = 1;
          dim.os = 1;
          _planForward = fftwf_plan_guru_split_dft_r2c(1, &dim, 0, 0, _data, _re, _im, FFTW_MEASURE);
          _planBackward = fftwf_plan_guru_split_dft_c2r(1, &dim, 0, 0, _re, _im, _data, FFTW_MEASURE);
        }
      }
    }

    virtual void fft(const float* data, float* re, float* im) override
    {
      ::memcpy(_data, data, _size * sizeof(float));
      fftwf_execute_split_dft_r2c(_planForward, _data, _re, _im);
      ::memcpy(re, _re, _complexSize * sizeof(float));
      ::memcpy(im, _im, _complexSize * sizeof(float));
    }

    virtual void ifft(float* data, const float* re, const float* im) override
    {
      ::memcpy(_re, re, _complexSize * sizeof(float));
      ::memcpy(_im, im, _complexSize * sizeof(float));
      fftwf_execute_split_dft_c2r(_planBackward, _re, _im, _data);
      detail::ScaleBuffer(data, _data, 1.0f / static_cast<float>(_size), _size);
    }

  private:
    size_t _size;
    size_t _complexSize;
    fftwf_plan _planForward;
    fftwf_plan _planBackward;
    float* _data;
    float* _re;
    float* _im;
  };


  /**
   * @internal
   * @brief Concrete FFT implementation
   */
  typedef FFTW3FFT AudioFFTImplementation;


#endif // AUDIOFFT_FFTW3_USED


  // =============================================================


  AudioFFT::AudioFFT() :
    _impl(new AudioFFTImplementation())
  {
  }


  AudioFFT::~AudioFFT()
  {
  }


  void AudioFFT::init(size_t size)
  {
    assert(detail::IsPowerOf2(size));
    _impl->init(size);
  }


  void AudioFFT::fft(const float* data, float* re, float* im)
  {
    _impl->fft(data, re, im);
  }


  void AudioFFT::ifft(float* data, const float* re, const float* im)
  {
    _impl->ifft(data, re, im);
  }


  size_t AudioFFT::ComplexSize(size_t size)
  {
    return (size / 2) + 1;
  }

} // End of namespace
