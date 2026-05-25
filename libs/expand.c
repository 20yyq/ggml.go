/*
 * @Author       : Eacher
 * @Date         : 2026-05-25 13:58:03
 * @LastEditTime : 2026-05-25 14:23:16
 * @LastEditors  : Eacher
 * --------------------------------------------------------------------------------<
 * @Description  : Please edit a descrition about this file at here.
 * --------------------------------------------------------------------------------<
 */
#include "expand.h"

#define CPU_NUMA_INIT "ggml_backend_cpu_numa_init"


void numa_init_fn(ggml_backend_reg_t reg, enum ggml_numa_strategy n)
{
	typedef void (*numa_func_t)(enum ggml_numa_strategy);
	numa_func_t func = (numa_func_t) ggml_backend_reg_get_proc_address(reg, CPU_NUMA_INIT);
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
