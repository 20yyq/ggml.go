// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 22:16:49
// @ LastEditTime : 2026-05-25 10:26:36
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #include <stdlib.h>
// #include "ggml.h"
// #include "ggml-backend.h"
// #include "gguf.h"
import "C"
import (
	"fmt"
)

func backend_init() {
	C.ggml_time_init()
	// needed to initialize f16 tables
	{
		var params C.struct_ggml_init_params
		var ctx *C.struct_ggml_context = C.ggml_init(params)
		C.ggml_free(ctx)
	}

	if C.ggml_backend_load_all(); C.ggml_backend_reg_count() == 0 {
		// hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function
		panic("no backends are loaded.")
	}
}

func backend_dinit() {
	C.ggml_quantize_free()
	fmt.Printf("backend_dinit \n")
}

type backends struct {
	_dev C.ggml_backend_dev_t
}
