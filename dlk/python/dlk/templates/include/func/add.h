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

#ifndef DLK_FUNC_ADD_H_INCLUDED
#define DLK_FUNC_ADD_H_INCLUDED

#include "global.h"


void func_Add(uint8_t *&input, float *&term, float *&output, int out_height, int out_width, int out_depth);

void func_Add(T_FLOAT input[], T_FLOAT term, T_FLOAT output[], T_UINT out_height, T_UINT out_width, T_UINT out_depth);

void func_Add(T_FLOAT input[], T_FLOAT term[], T_FLOAT output[], T_UINT out_height, T_UINT out_width, T_UINT out_depth);

void func_Add(T_FLOAT input[], T_FLOAT term, T_FLOAT output[], T_UINT out_depth);

void func_Add(T_FLOAT term, T_FLOAT input[], T_FLOAT output[], T_UINT out_height, T_UINT out_width, T_UINT out_depth);

void func_Add(T_FLOAT input[], T_FLOAT term[], T_FLOAT output[], T_UINT out_depth);

void func_Add_depthwise(T_FLOAT input[], T_FLOAT terms[], T_FLOAT output[], T_UINT out_height, T_UINT out_width, T_UINT out_depth);

void func_Add_depthwise(T_FLOAT input[], T_FLOAT terms[], T_FLOAT output[], T_UINT out_depth);

#endif // DLK_FUNC_ADD_H_INCLUDED
