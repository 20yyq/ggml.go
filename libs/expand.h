/*
 * @Author       : Eacher
 * @Date         : 2026-05-25 13:58:12
 * @LastEditTime : 2026-06-29 15:31:30
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

//
// ggml context OBJ copy
//

typedef struct {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
} _c_ggml_context_t;

extern void go_log_callback(enum ggml_log_level level, char * text, void * user_data);
extern void go_abort_callback(char * text);

void numa_init_fn(ggml_backend_reg_t, enum ggml_numa_strategy);
void set_n_threads(ggml_backend_reg_t, ggml_backend_t, int);
bool cpu_is_numa(ggml_backend_reg_t);
