// @@
// @ Author       : Eacher
// @ Date         : 2026-05-26 13:53:54
// @ LastEditTime : 2026-05-26 13:54:04
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #cgo CFLAGS: -I${SRCDIR}/../ggml/include
// #cgo CPPFLAGS: -I${SRCDIR}/../ggml/include
// #cgo LDFLAGS: -lstdc++ -lpthread
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml -lggml-base -lggml-cuda -lggml-cpu-x64
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml-cpu-alderlake -lggml-cpu-cannonlake -lggml-cpu-cascadelake
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml-cpu-cooperlake -lggml-cpu-haswell -lggml-cpu-icelake
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml-cpu-ivybridge -lggml-cpu-piledriver -lggml-cpu-sandybridge
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml-cpu-sapphirerapids -lggml-cpu-skylakex -lggml-cpu-sse42 -lggml-cpu-zen4
//
import "C"
