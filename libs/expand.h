/*
 * @Author       : Eacher
 * @Date         : 2026-05-25 13:58:12
 * @LastEditTime : 2026-05-25 14:26:16
 * @LastEditors  : Eacher
 * --------------------------------------------------------------------------------<
 * @Description  : Please edit a descrition about this file at here.
 * --------------------------------------------------------------------------------<
 */
#pragma once

#include <stdlib.h>
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

void numa_init_fn(ggml_backend_reg_t, enum ggml_numa_strategy);
