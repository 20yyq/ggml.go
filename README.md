<!--
 * @Author       : Eacher
 * @Date         : 2026-05-25 08:13:11
 * @LastEditTime : 2026-06-29 17:35:23
 * @LastEditors  : Eacher
 * --------------------------------------------------------------------------------<
 * @Description  : Please edit a descrition about this file at here.
 * --------------------------------------------------------------------------------<
-->
# ggml.go
Based on ggml's matrix algorithm interface library for the go language version.

## Compilation Steps.
```sh
	go generate cmd/build/main.go
	LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/libs/build/bin" go run test/main.go 
```

## Examples
```sh
	LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/libs/build/bin" go run examples/main.go simple
```