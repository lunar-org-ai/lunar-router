package weights

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// NpyArray represents a parsed NumPy array from an .npy file inside an .npz.
type NpyArray struct {
	Shape []int
	Dtype string // e.g. "<f8" (float64 LE), "<f4" (float32 LE)
	Data  []byte
}

// AsFloat64Flat returns the raw data as a flat []float64 slice.
// Supports float64 (<f8) and float32 (<f4) dtypes.
func (a *NpyArray) AsFloat64Flat() ([]float64, error) {
	switch a.Dtype {
	case "<f8":
		return a.decodeFloat64()
	case "<f4":
		return a.decodeFloat32AsFloat64()
	default:
		return nil, fmt.Errorf("unsupported dtype %q for float64 conversion", a.Dtype)
	}
}

func (a *NpyArray) decodeFloat64() ([]float64, error) {
	n := len(a.Data) / 8
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint64(a.Data[i*8 : (i+1)*8])
		result[i] = math.Float64frombits(bits)
	}
	return result, nil
}

func (a *NpyArray) decodeFloat32AsFloat64() ([]float64, error) {
	n := len(a.Data) / 4
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(a.Data[i*4 : (i+1)*4])
		result[i] = float64(math.Float32frombits(bits))
	}
	return result, nil
}

// AsFloat64_2D returns the data as a 2D slice (rows, cols).
// The array must have exactly 2 dimensions.
func (a *NpyArray) AsFloat64_2D() ([][]float64, error) {
	if len(a.Shape) != 2 {
		return nil, fmt.Errorf("expected 2D array, got %dD (shape %v)", len(a.Shape), a.Shape)
	}
	flat, err := a.AsFloat64Flat()
	if err != nil {
		return nil, err
	}
	rows, cols := a.Shape[0], a.Shape[1]
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = flat[i*cols : (i+1)*cols]
	}
	return result, nil
}

// AsString returns the data as a string (for scalar string arrays like "type").
// Handles both byte strings (|S*) and Unicode strings (<U*, >U*).
func (a *NpyArray) AsString() string {
	if strings.Contains(a.Dtype, "U") {
		// Unicode string: each character is 4 bytes (UTF-32)
		return decodeUTF32LE(a.Data)
	}
	// Byte string (|S*): raw bytes, null-padded
	s := string(a.Data)
	return strings.TrimRight(s, "\x00 ")
}

// decodeUTF32LE decodes a little-endian UTF-32 byte slice to a Go string.
func decodeUTF32LE(data []byte) string {
	var b strings.Builder
	for i := 0; i+3 < len(data); i += 4 {
		r := rune(binary.LittleEndian.Uint32(data[i : i+4]))
		if r == 0 {
			break
		}
		b.WriteRune(r)
	}
	return b.String()
}

// NpzFile represents a parsed .npz file (ZIP of .npy files).
type NpzFile struct {
	Arrays map[string]*NpyArray
}

// ReadNpz reads and parses a .npz file from disk.
func ReadNpz(path string) (*NpzFile, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open npz %s: %w", path, err)
	}
	defer r.Close()

	npz := &NpzFile{Arrays: make(map[string]*NpyArray)}

	for _, f := range r.File {
		name := f.Name
		// Remove .npy extension for the key
		name = strings.TrimSuffix(name, ".npy")

		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("open entry %s: %w", f.Name, err)
		}

		arr, err := parseNpy(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", f.Name, err)
		}

		npz.Arrays[name] = arr
	}

	return npz, nil
}

// Get returns an array by name, or nil if not found.
func (n *NpzFile) Get(name string) *NpyArray {
	return n.Arrays[name]
}

// parseNpy reads a single .npy formatted stream.
// Format: magic (\x93NUMPY), version (2 bytes), header_len (2 or 4 bytes), header (Python dict), data.
func parseNpy(r interface{ Read([]byte) (int, error) }) (*NpyArray, error) {
	// Read magic: \x93NUMPY
	magic := make([]byte, 6)
	if _, err := readFull(r, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic[0] != 0x93 || string(magic[1:6]) != "NUMPY" {
		return nil, fmt.Errorf("invalid npy magic: %x", magic)
	}

	// Read version
	version := make([]byte, 2)
	if _, err := readFull(r, version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}

	// Read header length (2 bytes for v1, 4 bytes for v2+)
	var headerLen int
	if version[0] == 1 {
		hl := make([]byte, 2)
		if _, err := readFull(r, hl); err != nil {
			return nil, fmt.Errorf("read header len: %w", err)
		}
		headerLen = int(binary.LittleEndian.Uint16(hl))
	} else {
		hl := make([]byte, 4)
		if _, err := readFull(r, hl); err != nil {
			return nil, fmt.Errorf("read header len: %w", err)
		}
		headerLen = int(binary.LittleEndian.Uint32(hl))
	}

	// Read header string (Python dict)
	headerBytes := make([]byte, headerLen)
	if _, err := readFull(r, headerBytes); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	header := string(headerBytes)

	// Parse dtype from header
	dtype, err := parseHeaderField(header, "descr")
	if err != nil {
		return nil, fmt.Errorf("parse descr: %w", err)
	}

	// Parse shape from header
	shape, err := parseShape(header)
	if err != nil {
		return nil, fmt.Errorf("parse shape: %w", err)
	}

	// Calculate data size
	elemSize := dtypeSize(dtype)
	if elemSize == 0 {
		// For string dtypes like |S6, |S11, etc.
		elemSize = parseStringDtypeSize(dtype)
	}
	totalElems := 1
	for _, s := range shape {
		totalElems *= s
	}
	dataSize := totalElems * elemSize

	// Read data
	data := make([]byte, dataSize)
	if _, err := readFull(r, data); err != nil {
		return nil, fmt.Errorf("read data (%d bytes): %w", dataSize, err)
	}

	return &NpyArray{
		Shape: shape,
		Dtype: dtype,
		Data:  data,
	}, nil
}

func readFull(r interface{ Read([]byte) (int, error) }, buf []byte) (int, error) {
	total := 0
	for total < len(buf) {
		n, err := r.Read(buf[total:])
		total += n
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

var descrRe = regexp.MustCompile(`'descr'\s*:\s*'([^']*)'`)
var shapeRe = regexp.MustCompile(`'shape'\s*:\s*\(([^)]*)\)`)

func parseHeaderField(header, field string) (string, error) {
	re := regexp.MustCompile(`'` + field + `'\s*:\s*'([^']*)'`)
	m := re.FindStringSubmatch(header)
	if m == nil {
		return "", fmt.Errorf("field %q not found in header: %s", field, header)
	}
	return m[1], nil
}

func parseShape(header string) ([]int, error) {
	m := shapeRe.FindStringSubmatch(header)
	if m == nil {
		return nil, fmt.Errorf("shape not found in header: %s", header)
	}
	parts := strings.Split(m[1], ",")
	var shape []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid shape element %q: %w", p, err)
		}
		shape = append(shape, n)
	}
	return shape, nil
}

func dtypeSize(dtype string) int {
	switch dtype {
	case "<f8", ">f8":
		return 8 // float64
	case "<f4", ">f4":
		return 4 // float32
	case "<i8", ">i8":
		return 8 // int64
	case "<i4", ">i4":
		return 4 // int32
	default:
		return 0
	}
}

// parseStringDtypeSize handles string dtypes like "|S6", "|S11".
func parseStringDtypeSize(dtype string) int {
	if strings.HasPrefix(dtype, "|S") {
		n, err := strconv.Atoi(dtype[2:])
		if err == nil {
			return n
		}
	}
	// Also handle Unicode strings |U<n> (4 bytes per char)
	if strings.HasPrefix(dtype, "<U") || strings.HasPrefix(dtype, "|U") {
		n, err := strconv.Atoi(dtype[2:])
		if err == nil {
			return n * 4
		}
	}
	return 1 // fallback
}
