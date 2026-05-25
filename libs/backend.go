// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 22:16:49
// @ LastEditTime : 2026-05-24 11:23:09
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

type backends struct {
	_dev C.ggml_backend_dev_t
}
