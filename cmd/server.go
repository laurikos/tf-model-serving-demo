package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"math"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const maxRequestBytes = 10 << 20
const maxImageBytes = 8 << 20

type predictor interface {
	Predict(context.Context, imageTensor) ([]float32, error)
	Ready(context.Context) error
}

type application struct {
	backend predictor
	labels  []string
	index   []byte
	busy    chan struct{}
}

type LabelResult struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}

type ClassifyResult struct {
	Filename string        `json:"filename"`
	Labels   []LabelResult `json:"labels"`
}

func newApplication(backend predictor, labels []string, index []byte) *application {
	return &application{backend: backend, labels: labels, index: index, busy: make(chan struct{}, 1)}
}

func (a *application) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("X-Content-Type-Options", "nosniff")
	switch r.URL.Path {
	case "/", "/index.html", "/healthz", "/readyz":
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			responseError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		switch r.URL.Path {
		case "/healthz":
			w.WriteHeader(http.StatusOK)
		case "/readyz":
			ctx, cancel := context.WithTimeout(r.Context(), time.Second)
			defer cancel()
			if err := a.backend.Ready(ctx); err != nil {
				responseError(w, "model unavailable", http.StatusServiceUnavailable)
				return
			}
			w.WriteHeader(http.StatusOK)
		default:
			http.ServeContent(w, r, "index.html", time.Time{}, bytes.NewReader(a.index))
		}
	case "/api/v1/inference":
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", "POST")
			responseError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		a.inference(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (a *application) inference(w http.ResponseWriter, r *http.Request) {
	// Acquire before reading the body so parallel uploads cannot multiply memory use.
	select {
	case a.busy <- struct{}{}:
		defer func() { <-a.busy }()
	default:
		w.Header().Set("Retry-After", "1")
		responseError(w, "another image is being processed; retry shortly", http.StatusTooManyRequests)
		return
	}
	if r.ContentLength > maxRequestBytes {
		responseError(w, "request exceeds 10 MiB", http.StatusRequestEntityTooLarge)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
	err := r.ParseMultipartForm(maxRequestBytes)
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	if err != nil {
		var sizeError *http.MaxBytesError
		if errors.As(err, &sizeError) {
			responseError(w, "request exceeds 10 MiB", http.StatusRequestEntityTooLarge)
		} else {
			responseError(w, "expected multipart form with one image", http.StatusBadRequest)
		}
		return
	}
	files := r.MultipartForm.File["image"]
	if len(files) != 1 {
		responseError(w, "provide exactly one image", http.StatusBadRequest)
		return
	}
	file := files[0]
	if file.Size > maxImageBytes {
		responseError(w, "image exceeds 8 MiB", http.StatusRequestEntityTooLarge)
		return
	}
	reader, err := file.Open()
	if err != nil {
		responseError(w, "could not read image", http.StatusBadRequest)
		return
	}
	defer reader.Close()
	data, err := io.ReadAll(io.LimitReader(reader, maxImageBytes+1))
	if err != nil {
		responseError(w, "could not read image", http.StatusBadRequest)
		return
	}
	if len(data) > maxImageBytes {
		responseError(w, "image exceeds 8 MiB", http.StatusRequestEntityTooLarge)
		return
	}
	input, err := normalizeImage(data)
	if err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, errImageTooLarge) {
			status = http.StatusRequestEntityTooLarge
		}
		responseError(w, err.Error(), status)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	probabilities, err := a.backend.Predict(ctx, input)
	if err != nil {
		log.Printf("inference failed: %v", err)
		responseError(w, "inference temporarily unavailable", http.StatusBadGateway)
		return
	}
	if len(probabilities) != len(a.labels) {
		responseError(w, "invalid model output", http.StatusBadGateway)
		return
	}
	labels := make([]LabelResult, len(a.labels))
	for i, p := range probabilities {
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) || p < 0 || p > 1 {
			responseError(w, "invalid model output", http.StatusBadGateway)
			return
		}
		labels[i] = LabelResult{Label: a.labels[i], Probability: p}
	}
	sort.SliceStable(labels, func(i, j int) bool { return labels[i].Probability > labels[j].Probability })
	name := filepath.Base(strings.ReplaceAll(file.Filename, "\\", "/"))
	responseJSON(w, http.StatusOK, ClassifyResult{Filename: name, Labels: labels})
}

func responseError(w http.ResponseWriter, message string, code int) {
	responseJSON(w, code, map[string]string{"error": message})
}

func responseJSON(w http.ResponseWriter, code int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Printf("response write failed: %v", err)
	}
}
