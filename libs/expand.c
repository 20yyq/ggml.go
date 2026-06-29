/*
 * @Author       : Eacher
 * @Date         : 2026-05-25 13:58:03
 * @LastEditTime : 2026-06-27 13:57:20
 * @LastEditors  : Eacher
 * --------------------------------------------------------------------------------<
 * @Description  : Please edit a descrition about this file at here.
 * --------------------------------------------------------------------------------<
 */
#include "expand.h"

#define S_CPU_NUMA_INIT "ggml_backend_cpu_numa_init"
#define S_SET_N_THREADS "ggml_backend_set_n_threads"
#define S_CPU_IS_NUMA "ggml_backend_cpu_is_numa"


void numa_init_fn(ggml_backend_reg_t reg, enum ggml_numa_strategy n)
{
	typedef void (*numa_func_t)(enum ggml_numa_strategy);
	numa_func_t func = (numa_func_t) ggml_backend_reg_get_proc_address(reg, S_CPU_NUMA_INIT);
	if (!func) return;

	switch (n) {
	case GGML_NUMA_STRATEGY_DISABLED:
	case GGML_NUMA_STRATEGY_ISOLATE:
	case GGML_NUMA_STRATEGY_NUMACTL:
	case GGML_NUMA_STRATEGY_MIRROR:
	case GGML_NUMA_STRATEGY_COUNT: {
		func(n);
	}break;
	case GGML_NUMA_STRATEGY_DISTRIBUTE:
		break;
	}
}

void set_n_threads(ggml_backend_reg_t reg, ggml_backend_t backend, int n)
{
	ggml_backend_set_n_threads_t func = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, S_SET_N_THREADS);
	if (!func) return;
	func(backend,n);
}

bool cpu_is_numa(ggml_backend_reg_t reg)
{
	typedef bool (*func_t)(void);
	func_t func = (func_t) ggml_backend_reg_get_proc_address(reg, S_CPU_IS_NUMA);
	if (!func) return false;
	return func();
}
