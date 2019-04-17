/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cassert>
#include <cstring>

#include "global.h"
#include "func/impl/quantized_conv2d_kn2row.h"
#include "matrix_view.h"
#include "matrix/quantized_multiplication.h"
#include "matrix/shift_add.h"
#include "pack_input_to_qwords.h"
#include "time_measurement.h"

namespace {

// kernel format converter
// ohwi : oc kh kw ic, hwoi: kh kw oc ic
void quantized_ohwi_to_hwoi(const QUANTIZED_PACKED_KERNEL ohwi[], QUANTIZED_PACKED_KERNEL hwoi[], const struct binary_convolution_parameters& p) {
   Measurement::Start("quantized_ohwi_to_hwoi");

   int ic = p.normal_conv_params.kernel_depth / 32;
   int oc = p.normal_conv_params.output_channels;
   int kh = p.normal_conv_params.kernel_height;
   int kw = p.normal_conv_params.kernel_width;

   for (unsigned int i = 0; i < kh*kw; ++i) {
     for (unsigned int j = 0; j < oc; ++j) {
       for (unsigned int k = 0; k < ic; ++k) {
         hwoi[i*oc*ic + j*ic + k] = ohwi[i*ic + j*ic*kh*kw + k];
       }
     }
   }

   Measurement::Stop();
 }

void ApplyThresholds(
    dlk::MatrixView<BIN_CONV_OUTPUT, dlk::MatrixOrder::ColMajor> &result,
    const binary_convolution_parameters &p) {
  Measurement::Start("ApplyThresholds");
  T_INT ts [NUM_OF_A2W1_THRESHOLD-1];
  for (unsigned int i = 0; i < result.rows(); ++i) {
    for (unsigned int j = 0; j < result.cols(); ++j) {
      BIN_CONV_OUTPUT d = *result.data(i, j);
      for(int k = 0;k < NUM_OF_A2W1_THRESHOLD - 1; k++){
        ts[k] = p.thresholds[NUM_OF_A2W1_THRESHOLD * i+k];
      }
      T_INT flag = p.thresholds[NUM_OF_A2W1_THRESHOLD * i + (NUM_OF_A2W1_THRESHOLD -1)];
      BIN_CONV_OUTPUT new_d;

      if (flag == 1) { // increasing function
        for(int cont = 0; cont < NUM_OF_A2W1_THRESHOLD - 1  ; cont++){
          if( d < ts[cont]){
            new_d = cont;
            break;
          }
        }
        if (d >= ts[NUM_OF_A2W1_THRESHOLD-2])
          new_d = NUM_OF_A2W1_THRESHOLD-1;

      }else if (flag == -1) { // decreasing function
         for(int cont = NUM_OF_A2W1_THRESHOLD - 2; cont >= 0  ; cont--){
            if( d > ts[cont]){
            new_d = cont - (NUM_OF_A2W1_THRESHOLD - 2) ;
            break;
          }
        }
        if (d <= ts[0])
          new_d = NUM_OF_A2W1_THRESHOLD-1;
      }else {
        new_d = flag - 2;                 // note: 2 is a magic number!
        assert(0 <= new_d && new_d <= 2); // unsinged 2bits
      }
      *result.data(i, j) = new_d;
    }
  }

  Measurement::Stop();
}

} // namespace

namespace dlk {

namespace impl {

void QuantizedConv2DKn2Row(QUANTIZED_NOT_PACKED input[],
                                  const QUANTIZED_PACKED_KERNEL kernel[],
                                  const binary_convolution_parameters &p) {
  using namespace dlk;

  int ic = p.normal_conv_params.kernel_depth;
  int ih = p.normal_conv_params.input_height;
  int iw = p.normal_conv_params.input_width;
  int oc = p.normal_conv_params.output_channels;
  int oh = p.normal_conv_params.output_height;
  int ow = p.normal_conv_params.output_width;
  int kh = p.normal_conv_params.kernel_height;
  int kw = p.normal_conv_params.kernel_width;

  assert(ih * iw == oh * ow);
  assert(MAX_SIZE_IM2COL_INPUTS_PER_LAYER >= ic * kh * kw * ih * iw);

  Measurement::Start("quantized-kn2row");

  int kernel_buf_size = kh * kw * ic * oc / 32;
  auto kernel_hwoi = new QUANTIZED_PACKED_KERNEL[kernel_buf_size]();
  quantized_ohwi_to_hwoi(kernel, kernel_hwoi, p);

  pack_input_to_qwords(input, p.device_input_buf, ih * iw * ic, 2);
  auto kernel_ = MatrixView<QUANTIZED_PACKED_KERNEL, MatrixOrder::RowMajor>(
      kernel_hwoi, oc * kh * kw, ic / 32);
  auto input_ = MatrixView<QUANTIZED_PACKED, MatrixOrder::ColMajor>(
      p.device_input_buf, ic / 16, ih * iw);
  auto output_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
      p.device_output_buf, oc, ih * iw);
  printf("kw = %d", kw);
  if (kh == kw && kw == 3) {
    unsigned bufsize = oc * kh * kw * ih * iw;
    BIN_CONV_OUTPUT *kn2row_buf = new BIN_CONV_OUTPUT[bufsize]();
    std::memset(kn2row_buf, 0, bufsize);
    auto buf_ = MatrixView<BIN_CONV_OUTPUT, MatrixOrder::ColMajor>(
        kn2row_buf, oc * kh * kw, ih * iw);

    quantized_matrix_multiplication(kernel_, input_, buf_);
    matrix_shift_add(buf_, output_, p.normal_conv_params);
    delete[] kn2row_buf;
  } else if (kh == kw && kw == 1) {
    quantized_matrix_multiplication(kernel_, input_, output_);
  } else {
    std::cerr << "Only 1x1 or 3x3 convolutions are supported." << std::endl;
    assert(false);
  }

  delete[] kernel_hwoi;

  if (p.thresholds != nullptr) {
    ApplyThresholds(output_, p);
  }

  Measurement::Stop();
}

} // namespace impl

} // namespace dlk
