// @@
// @ Author       : Eacher
// @ Date         : 2026-06-29 13:33:09
// @ LastEditTime : 2026-06-30 08:14:18
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package simple

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"unsafe"

	ggmlgo "ggml.go"
	"ggml.go/libs"
)

var matrix_A = []float32{
	2, 8,
	5, 1,
	4, 2,
	8, 6,
}

var matrix_B = []float32{
	10, 5,
	9, 9,
	5, 4,
}

type simple_model struct {
	a, b   libs.Tensor
	ggml   libs.GGML
	sched  libs.Sched
	ctx    context.Context
	cancel context.CancelCauseFunc
}

func Main() error {
	libs.Init(libs.InitParams{})
	fmt.Println("version:", libs.LIB_ggml_version(), libs.LIB_ggml_commit())
	obj := simple_model{}
	obj.ctx, obj.cancel = context.WithCancelCause(context.Background())
	err := obj.ggml.Init(2048, true, obj.ctx)
	defer obj.cancel(io.EOF)
	if err != nil {
		return err
	}
	go func() {
		<-obj.ctx.Done()
		fmt.Printf("Main close \n")
	}()
	l := libs.GetDevs()
	var ls []*libs.Backend
	for k, v := range l {
		ls = append(ls, &libs.Backend{Dev: v})
		if err = ls[k].Init(obj.ctx); err != nil {
			return err
		}
	}

	if err = obj.sched.Init(ls, 0, obj.ctx); err != nil {
		return err
	}

	// build_graph
	// result = a*b^T
	{
		// create tensors
		info := libs.TensorInfo{T: ggmlgo.GGML_TYPE_F32, NE: [4]int64{2, 4, 1, 1}}
		err = obj.a.Init(&obj.ggml, 0, &info)
		if err != nil {
			return err
		}
		info.NE[1] = 3
		err = obj.b.Init(&obj.ggml, 1, &info)
		if err != nil {
			return err
		}

		result, err := obj.a.MUL_MAT(obj.b)
		if err != nil {
			return err
		}
		result.ForwardExpand()
	}

	// compute
	{
		obj.sched.Rest()
		obj.sched.AllocGraph(&obj.ggml)

		var br []byte
		br = unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(matrix_A))), len(matrix_A)*int(unsafe.Sizeof(matrix_A[0])))
		obj.a.SetData(br)
		br = unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(matrix_B))), len(matrix_B)*int(unsafe.Sizeof(matrix_B[0])))
		obj.b.SetData(br)
		var data libs.ResultTensor
		data, err = obj.sched.GraphCompute(&obj.ggml)
		fdata := make([]float32, len(data.Data)/int(libs.LIB_ggml_type_size(data.Info.T)))
		binary.Read(bytes.NewReader(data.Data), binary.LittleEndian, &fdata)
		fmt.Printf("fdata %f  %v \n", fdata, data.Info)
		// fdata [60.000000 55.000000 50.000000 110.000000 90.000000 54.000000 54.000000 126.000000 42.000000 29.000000 28.000000 64.000000]
	}
	return nil
}
