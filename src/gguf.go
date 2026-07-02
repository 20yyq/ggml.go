// @@
// @ Author       : Eacher
// @ Date         : 2026-06-22 10:51:45
// @ LastEditTime : 2026-06-30 14:28:29
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package src

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	ggmlgo "ggml.go"
	"ggml.go/libs"
)

const GGML_MAX_NAME uint64 = 64
const GGML_MAX_DIMS uint64 = 4
const INT64_MAX int64 = 0x7FFFFFFFFFFFFFFF
const SIZE_MAX uint64 = 0xFFFFFFFFFFFFFFFF

const GGUF_MAX_STRING_LENGTH uint64 = 1024 * 1024 * 1024
const GGUF_MAX_ARRAY_ELEMENTS uint64 = 1024 * 1024 * 1024

type gguf_kv[T value_t | string] struct {
	tp   ggmlgo.GGUF_TYPE
	list []T
}

func (p *gguf_kv[T]) Println() {
	fmt.Println("---------", p.list[0])
}

func (p *gguf_kv[T]) GetType() ggmlgo.GGUF_TYPE {
	return p.tp
}

type GGUF_KV interface {
	Println()
	GetType() ggmlgo.GGUF_TYPE
}

func GetList[T value_t | string](obj GGUF_KV) []T {
	ptr, ok := obj.(*gguf_kv[T])
	if !ok {
		return nil
	}
	return ptr.list
}

type GGUF struct {
	org                     libs.GGML
	file                    string
	version                 uint32
	ctx_alignment, ctx_size uint64
	n_kv, n_tensors         int64
	ctx                     context.Context
	cancel                  context.CancelCauseFunc
}

func (ptr *GGUF) Init(file string, kvMaps *map[string]GGUF_KV, tensorMaps *map[string]*libs.Tensor, ctx context.Context) error {
	if kvMaps == nil || tensorMaps == nil {
		return errors.New("map ptr == nil")
	}
	f, err := os.Open(file)
	if err != nil {
		return err
	}
	defer f.Close()
	b := [32]byte{}
	if _, err = f.Read(b[:24]); err != nil {
		return err
	}
	read := bytes.NewBuffer(b[:])
	if !ptr.magicAndVersion(read) {
		return err
	}

	// header
	// kvAndTensors
	{
		if !binaryRead(read, &ptr.n_tensors) {
			return err
		}
		if !binaryRead(read, &ptr.n_kv) {
			return err
		}
	}

	ptr.ctx, ptr.cancel = context.WithCancelCause(ctx)

	if err = ptr.org.Init(uint64(ptr.n_tensors), false, ptr.ctx); err != nil {
		ptr.cancel(io.EOF)
		return err
	}
	go ptr.done()

	ptr.ctx_alignment, ptr.file = 32, file

	if !ptr.kvPairs(f, ptr.n_kv, kvMaps) {
		ptr.cancel(io.EOF)
		return err
	}

	if !ptr.tensorPairs(f, ptr.n_tensors, tensorMaps) {
		ptr.cancel(io.EOF)
		return err
	}
	return nil
}

func (ptr *GGUF) Close() error {
	ptr.cancel(io.EOF)
	return nil
}

func (ptr *GGUF) Done() <-chan struct{} {
	return ptr.ctx.Done()
}

func (ptr *GGUF) done() {
	<-ptr.ctx.Done()
	ptr.close()
}

func (ptr *GGUF) close() error {
	return ptr.org.Close()
}

func (ptr *GGUF) magicAndVersion(r io.Reader) bool {
	ok := true
	b := [4]byte{}
	_, err := r.Read(b[:])
	if err != nil || "GGUF" != string(b[:]) {
		return false
	}
	binaryRead(r, &ptr.version)
	return ok
}

func (ptr *GGUF) kvPairs(r io.Reader, n_kv int64, kvMaps *map[string]GGUF_KV) bool {
	res := true
	*kvMaps = map[string]GGUF_KV{}
	for i := int64(0); res && i < n_kv; i++ {
		ok, key, t, n := true, "", int32(-1), uint64(1)
		if ok = readString(r, &key); !ok {
			res = false
			break
		}
		if _, ok = (*kvMaps)[key]; ok {
			fmt.Println("duplicate key ", key)
			res = false
			break
		}
		(*kvMaps)[key] = nil
		if ok = binaryRead(r, &t); !ok {
			res = false
			break
		}
		if ggmlgo.GGUF_TYPE(t) == ggmlgo.GGUF_TYPE_ARRAY {
			if ok = binaryRead(r, &t); !ok {
				res = false
				break
			}
			if ok = binaryRead(r, &n); !ok {
				res = false
				break
			}
			if n > GGUF_MAX_ARRAY_ELEMENTS {
				res = false
				break
			}
		}
		// fmt.Println(key, n, t)
		switch ggmlgo.GGUF_TYPE(t) {
		case ggmlgo.GGUF_TYPE_INT8:
			if n > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[int8]{tp: ggmlgo.GGUF_TYPE(t), list: make([]int8, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				return false
			}
		case ggmlgo.GGUF_TYPE_UINT8:
			if n > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[uint8]{tp: ggmlgo.GGUF_TYPE(t), list: make([]uint8, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_INT16:
			if n*2 > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[int16]{tp: ggmlgo.GGUF_TYPE(t), list: make([]int16, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_UINT16:
			if n*2 > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[uint16]{tp: ggmlgo.GGUF_TYPE(t), list: make([]uint16, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_INT32:
			if n*4 > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[int32]{tp: ggmlgo.GGUF_TYPE(t), list: make([]int32, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_UINT32:
			if n*4 > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[uint32]{tp: ggmlgo.GGUF_TYPE(t), list: make([]uint32, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
			if key == "general.alignment" {
				ptr.ctx_alignment = uint64(value.list[0])
			}
		case ggmlgo.GGUF_TYPE_INT64:
			if n*8 > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[int64]{tp: ggmlgo.GGUF_TYPE(t), list: make([]int64, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_UINT64:
			if n*8 > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[uint64]{tp: ggmlgo.GGUF_TYPE(t), list: make([]uint64, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_BOOL:
			if n > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[bool]{tp: ggmlgo.GGUF_TYPE(t), list: make([]bool, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_FLOAT32:
			if n*4 > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[float32]{tp: ggmlgo.GGUF_TYPE(t), list: make([]float32, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_FLOAT64:
			if n*8 > SIZE_MAX {
				res = false
				break
			}
			value := &gguf_kv[float64]{tp: ggmlgo.GGUF_TYPE(t), list: make([]float64, n)}
			(*kvMaps)[key] = value
			if ok = binaryReadSlice(r, value.list); !ok {
				res = false
				break
			}
		case ggmlgo.GGUF_TYPE_STRING:
			value := &gguf_kv[string]{tp: ggmlgo.GGUF_TYPE(t), list: make([]string, n)}
			(*kvMaps)[key] = value
			for i := uint64(0); i < n; i++ {
				if ok = readString(r, &value.list[i]); !ok {
					res = false
					break
				}
			}
		case ggmlgo.GGUF_TYPE_ARRAY:
			fallthrough
		default:
			fmt.Println("duplicate array ", key)
			res = false
			break
		}
	}
	if int(n_kv) != len((*kvMaps)) {
		fmt.Println("failed to read key-value pairs int(n_kv) != len(kvMaps)")
		res = false
	}

	if ptr.ctx_alignment == 0 || (ptr.ctx_alignment&(ptr.ctx_alignment-1)) != 0 {
		fmt.Println("alignment is not a power of 2", ptr.ctx_alignment)
		res = false
	}

	return res
}

func (ptr *GGUF) tensorPairs(r io.Reader, n_tensors int64, tensorMaps *map[string]*libs.Tensor) bool {
	*tensorMaps = map[string]*libs.Tensor{}
	for i := int64(0); i < n_tensors; i++ {
		ok, obj, n_dims, org := true, &libs.TensorInfo{NE: [4]int64{1, 1, 1, 1}}, uint32(0), &libs.Tensor{}
		if ok = readString(r, &obj.Name); !ok {
			return false
		}
		if _, ok = (*tensorMaps)[obj.Name]; ok {
			fmt.Println("duplicate name ", obj.Name)
			return false
		}
		(*tensorMaps)[obj.Name] = org
		// tensor shape
		{
			if ok = binaryRead(r, &n_dims); !ok {
				return false
			}
			if uint64(n_dims) > GGML_MAX_DIMS {
				fmt.Println(obj.Name, ": tensor has invalid number of dimensions:", n_dims, " > ", GGML_MAX_DIMS)
				return false
			}

			if ok = binaryReadSlice(r, obj.NE[:n_dims]); !ok {
				return false
			}
			if obj.NE[0] < 0 || obj.NE[1] < 0 || obj.NE[2] < 0 || obj.NE[3] < 0 {
				fmt.Println(obj.Name, ": tensor has invalid number of elements:", obj.NE[0], obj.NE[1], obj.NE[2], obj.NE[3])
				return false

			}

			// check that the total number of elements is representable
			if ok && (INT64_MAX/obj.NE[1] <= obj.NE[0]) || (INT64_MAX/obj.NE[2] <= obj.NE[0]*obj.NE[1]) || (INT64_MAX/obj.NE[3] <= obj.NE[0]*obj.NE[1]*obj.NE[2]) {
				fmt.Println(obj.Name, ": total number of elements in tensor with shape ", obj.NE[0], obj.NE[1], obj.NE[2], obj.NE[3])
				return false
			}
		}

		// tensor type
		{
			if ok = binaryRead(r, &n_dims); !ok {
				return false
			}

			// check that tensor type is within defined range
			if obj.T = ggmlgo.GGML_TYPE(n_dims); obj.T >= ggmlgo.GGML_TYPE_COUNT {
				fmt.Println(obj.Name, ": tensor has invalid ggml type:", obj.T, ggmlgo.GGML_TYPE_COUNT)
				return false
			}
		}

		// tensor data offset within buffer
		if ok = binaryRead(r, &obj.Offset); !ok {
			return false
		}

		// checker offset
		// compute the total size of the data section, taking into account the alignment
		if obj.Offset != ptr.ctx_size {
			fmt.Println(obj.Name, i, ": tensor has offset %", obj.Offset, ", expected %", ptr.ctx_size)
			return false
		}

		if err := org.Init(&ptr.org, int(i), obj); err != nil {
			fmt.Println(obj.Name, ": tensor init err", err)
			delete((*tensorMaps), obj.Name)
			return false
		}

		padded_size := libs.LIB_ggml_padding(obj.P_nbytes(), ptr.ctx_alignment)
		if SIZE_MAX-ptr.ctx_size < padded_size {
			fmt.Println(obj.Name, ": tensor size overflow, cannot accumulate size", ptr.ctx_size, padded_size)
			return false
		}

		ptr.ctx_size += padded_size
	}
	return true
}
