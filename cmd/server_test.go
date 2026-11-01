package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

type fakePredictor struct {
	call     func(context.Context, imageTensor) ([]float32, error)
	readyErr error
}

func (f fakePredictor) Predict(ctx context.Context, image imageTensor) ([]float32, error) {
	if f.call != nil {
		return f.call(ctx, image)
	}
	return []float32{0.2, 0.8}, nil
}
func (f fakePredictor) Ready(context.Context) error { return f.readyErr }

func fixture(t *testing.T, format string) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	for y := range 2 {
		for x := range 2 {
			img.SetNRGBA(x, y, color.NRGBA{R: 128, G: 64, B: 32, A: 255})
		}
	}
	var buf bytes.Buffer
	var err error
	if format == "png" {
		err = png.Encode(&buf, img)
	} else {
		err = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95})
	}
	if err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func upload(t *testing.T, name string, data []byte) *http.Request {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	part, err := w.CreateFormFile("image", name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = part.Write(data); err != nil {
		t.Fatal(err)
	}
	if err = w.Close(); err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/api/v1/inference", &buf)
	r.Header.Set("Content-Type", w.FormDataContentType())
	return r
}

func TestUploadUsesImageContents(t *testing.T) {
	for _, format := range []string{"png", "jpeg"} {
		t.Run(format, func(t *testing.T) {
			var calls int
			backend := fakePredictor{call: func(_ context.Context, input imageTensor) ([]float32, error) {
				calls++
				if len(input) != 299 || len(input[0]) != 299 {
					t.Fatal("incorrect tensor shape")
				}
				if math.Abs(float64(input[0][0][0])-128.0/255) > 0.02 {
					t.Fatal("incorrect normalization")
				}
				return []float32{0.2, 0.8}, nil
			}}
			a := newApplication(backend, []string{"hamburger", "not hamburger"}, nil)
			w := httptest.NewRecorder()
			a.ServeHTTP(w, upload(t, "wrong.extension", fixture(t, format)))
			if w.Code != http.StatusOK || calls != 1 {
				t.Fatalf("status=%d calls=%d body=%s", w.Code, calls, w.Body.String())
			}
			var result ClassifyResult
			if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.Filename != "wrong.extension" || len(result.Labels) != 2 || result.Labels[0].Label != "not hamburger" {
				t.Fatalf("unexpected response: %+v", result)
			}
		})
	}
}

func TestInvalidUploadsNeverReachModel(t *testing.T) {
	dimensionBomb := fixture(t, "png")
	binary.BigEndian.PutUint32(dimensionBomb[16:20], 8193)
	binary.BigEndian.PutUint32(dimensionBomb[29:33], crc32.ChecksumIEEE(dimensionBomb[12:29]))
	oversize := upload(t, "large.png", bytes.Repeat([]byte{'x'}, maxImageBytes+1))
	chunkedOversize := upload(t, "large.png", bytes.Repeat([]byte{'x'}, maxRequestBytes+1))
	chunkedOversize.ContentLength = -1
	var calls int
	a := newApplication(fakePredictor{call: func(context.Context, imageTensor) ([]float32, error) { calls++; return nil, nil }}, []string{"a", "b"}, nil)
	for _, tc := range []struct {
		name    string
		request *http.Request
		status  int
	}{
		{"missing multipart", httptest.NewRequest("POST", "/api/v1/inference", nil), 400},
		{"invalid image", upload(t, "a.png", []byte("not an image")), 400},
		{"truncated image", upload(t, "a.jpg", fixture(t, "jpeg")[:40]), 400},
		{"oversize file", oversize, 413},
		{"oversize chunked body", chunkedOversize, 413},
		{"dimension bomb", upload(t, "a.png", dimensionBomb), 413},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			a.ServeHTTP(w, tc.request)
			if w.Code != tc.status {
				t.Fatalf("status=%d want=%d body=%s", w.Code, tc.status, w.Body.String())
			}
		})
	}
	if calls != 0 {
		t.Fatalf("model called %d times", calls)
	}
}

func TestInferenceFailureDoesNotStopService(t *testing.T) {
	count := 0
	a := newApplication(fakePredictor{call: func(context.Context, imageTensor) ([]float32, error) {
		count++
		if count == 1 {
			return nil, errors.New("native model failed")
		}
		return []float32{0.9, 0.1}, nil
	}}, []string{"a", "b"}, nil)
	for _, status := range []int{502, 200} {
		w := httptest.NewRecorder()
		a.ServeHTTP(w, upload(t, "a.png", fixture(t, "png")))
		if w.Code != status {
			t.Fatalf("status=%d want=%d", w.Code, status)
		}
	}
}

func TestConcurrentInferenceIsRejected(t *testing.T) {
	started, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	a := newApplication(fakePredictor{call: func(context.Context, imageTensor) ([]float32, error) {
		calls.Add(1)
		close(started)
		<-release
		return []float32{0.5, 0.5}, nil
	}}, []string{"a", "b"}, nil)
	first, second := upload(t, "a.png", fixture(t, "png")), upload(t, "b.png", fixture(t, "png"))
	go func() { defer close(done); a.ServeHTTP(httptest.NewRecorder(), first) }()
	<-started
	w := httptest.NewRecorder()
	a.ServeHTTP(w, second)
	close(release)
	<-done
	if w.Code != 429 || w.Header().Get("Retry-After") != "1" || calls.Load() != 1 {
		t.Fatalf("status=%d calls=%d", w.Code, calls.Load())
	}
}

func TestRoutesAndReadiness(t *testing.T) {
	a := newApplication(fakePredictor{readyErr: errors.New("offline")}, nil, []byte("<h1>Upload</h1>"))
	for _, tc := range []struct {
		method, path string
		status       int
	}{
		{"GET", "/", 200}, {"GET", "/healthz", 200}, {"GET", "/readyz", 503},
		{"GET", "/models/myfirstmodel/2/saved_model.pb", 404}, {"POST", "/", 405},
		{"GET", "/api/v1/inference", 405},
	} {
		w := httptest.NewRecorder()
		a.ServeHTTP(w, httptest.NewRequest(tc.method, tc.path, nil))
		if w.Code != tc.status {
			t.Errorf("%s %s: %d", tc.method, tc.path, w.Code)
		}
	}
	for _, output := range [][]float32{nil, {1}, {float32(math.NaN()), 0}, {2, -1}} {
		a.labels = []string{"a", "b"}
		a.backend = fakePredictor{call: func(context.Context, imageTensor) ([]float32, error) { return output, nil }}
		w := httptest.NewRecorder()
		a.ServeHTTP(w, upload(t, "test.png", fixture(t, "png")))
		if w.Code != 502 {
			t.Errorf("invalid model output %v: HTTP %d", output, w.Code)
		}
	}
}

func TestBilinearCoordinatesAndTransparentRGB(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	img.SetNRGBA(0, 0, color.NRGBA{R: 255, A: 0})
	img.SetNRGBA(1, 0, color.NRGBA{G: 255, A: 255})
	img.SetNRGBA(0, 1, color.NRGBA{B: 255, A: 255})
	img.SetNRGBA(1, 1, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	input, err := normalizeImage(b.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if input[0][0] != [3]float32{1, 0, 0} {
		t.Fatalf("alpha changed RGB: %v", input[0][0])
	}
	x, y := float32(74)*float32(2)/299, float32(37)*float32(2)/299
	expect := [3]float32{1 - x - y + 2*x*y, x, y}
	for c, want := range expect {
		if math.Abs(float64(input[37][74][c]-want)) > 1e-6 {
			t.Fatalf("channel %d=%g want %g", c, input[37][74][c], want)
		}
	}
	if input[298][298] != [3]float32{1, 1, 1} {
		t.Fatal("edge was not clamped")
	}
}

func TestServerListenFailure(t *testing.T) {
	if err := run(context.Background(), &http.Server{Addr: "invalid"}); err == nil || !strings.Contains(err.Error(), "address") {
		t.Fatalf("error=%v", err)
	}
}
