// @@
// @ Author       : Eacher
// @ Date         : 2026-06-22 11:12:52
// @ LastEditTime : 2026-06-24 14:58:56
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package src

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

var binaryIface binary.ByteOrder = binary.LittleEndian

type value_t interface {
	bool | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

func binaryRead[T value_t](r io.Reader, ptr *T) bool {
	b, ok := make([]T, 1), false
	if binaryReadSlice(r, b) {
		*ptr, ok = b[0], true
	}
	return ok
}

func binaryReadSlice[T value_t](r io.Reader, ptr []T) bool {
	if ptr == nil {
		return false
	}

	b := make([]byte, len(ptr))
	switch v := any(ptr).(type) {
	case []bool:
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		for k, v1 := range b {
			v[k] = (v1 != 0)
		}
	case []int8:
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		for k, v1 := range b {
			v[k] = int8(v1)
		}
	case []uint8:
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		for k, v1 := range b {
			v[k] = uint8(v1)
		}
	case []int16:
		b = append(b, make([]byte, len(v))...)
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		idx, start := 0, 0
		for i := 0; i < len(b); {
			i += 2
			v[idx] = int16(binaryIface.Uint16(b[start:i]))
			start = i
			idx++
		}
	case []uint16:
		b = append(b, make([]byte, len(v))...)
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		idx, start := 0, 0
		for i := 0; i < len(b); {
			i += 2
			v[idx] = binaryIface.Uint16(b[start:i])
			start = i
			idx++
		}
	case []int32:
		b = append(b, make([]byte, len(v)*3)...)
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		idx, start := 0, 0
		for i := 0; i < len(b); {
			i += 4
			v[idx] = int32(binaryIface.Uint32(b[start:i]))
			start = i
			idx++
		}
	case []uint32:
		b = append(b, make([]byte, len(v)*3)...)
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		idx, start := 0, 0
		for i := 0; i < len(b); {
			i += 4
			v[idx] = binaryIface.Uint32(b[start:i])
			start = i
			idx++
		}
	case []int64:
		b = append(b, make([]byte, len(v)*7)...)
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		idx, start := 0, 0
		for i := 0; i < len(b); {
			i += 8
			v[idx] = int64(binaryIface.Uint64(b[start:i]))
			start = i
			idx++
		}
	case []uint64:
		b = append(b, make([]byte, len(v)*7)...)
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		idx, start := 0, 0
		for i := 0; i < len(b); {
			i += 8
			v[idx] = binaryIface.Uint64(b[start:i])
			start = i
			idx++
		}
	case []float32:
		b = append(b, make([]byte, len(v)*3)...)
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		idx, start := 0, 0
		for i := 0; i < len(b); {
			i += 4
			v[idx] = math.Float32frombits(binaryIface.Uint32(b[start:i]))
			start = i
			idx++
		}
	case []float64:
		b = append(b, make([]byte, len(v)*7)...)
		if n, err := r.Read(b); err != nil {
			fmt.Println("read: ", n, err, b)
			return false
		}
		idx, start := 0, 0
		for i := 0; i < len(b); {
			i += 8
			v[idx] = math.Float64frombits(binaryIface.Uint64(b[start:i]))
			start = i
			idx++
		}
	default:
		fmt.Println("default: type err")
		return false
	}
	return true
}

func readString(r io.Reader, str *string) bool {
	ok := true
	len := uint64(0)
	if ok = binaryRead(r, &len); !ok {
		return ok
	}
	if len > GGUF_MAX_STRING_LENGTH {
		fmt.Println("string length :", len, " exceeds maximum", GGUF_MAX_STRING_LENGTH)
		return false
	}
	// if (size > nbytes_remain) {
	// 	GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
	// 	return false;
	// }
	b := make([]byte, len)
	n, err := r.Read(b)
	if err != nil {
		return false
	}
	if ok = (n == int(len)); !ok {
		return ok
	}
	*str = string(b)
	return ok
}
