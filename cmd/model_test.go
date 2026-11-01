package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestModelClientContract(t *testing.T) {
	for _, response := range []string{`{"outputs":[[0.1,0.9]]}`, `{"outputs":{"prediction":[[0.1,0.9]]}}`} {
		backend := &modelClient{url: "http://127.0.0.1:8501/v1/models/hamppa/versions/2", client: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.URL.Path != "/v1/models/hamppa/versions/2:predict" || r.Method != "POST" {
				t.Fatalf("unexpected request: %s %s", r.Method, r.URL)
			}
			var request struct {
				Signature string                   `json:"signature_name"`
				Inputs    map[string][]imageTensor `json:"inputs"`
			}
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatal(err)
			}
			if request.Signature != "serving_default" || len(request.Inputs["image"]) != 1 || request.Inputs["image"][0][0][0][0] != 0.25 {
				t.Fatal("incorrect tensor request")
			}
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(response)), Header: make(http.Header)}, nil
		})}}
		output, err := backend.Predict(context.Background(), imageTensor{{{0.25, 0.5, 1}}})
		if err != nil || len(output) != 2 || output[1] != 0.9 {
			t.Fatalf("output=%v error=%v", output, err)
		}
	}
}

func TestModelClientRejectsBadResponses(t *testing.T) {
	for _, tc := range []struct {
		status int
		body   string
	}{
		{503, `{"error":"unavailable"}`}, {200, `{"outputs":[]}`}, {200, `{"outputs":[[0,1],[1,0]]}`},
		{200, "not JSON"}, {200, strings.Repeat("x", (64<<10)+1)},
	} {
		backend := &modelClient{url: "http://127.0.0.1/model", client: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: tc.status, Body: io.NopCloser(strings.NewReader(tc.body))}, nil
		})}}
		if _, err := backend.Predict(context.Background(), imageTensor{}); err == nil {
			t.Errorf("accepted %d %s", tc.status, tc.body[:min(20, len(tc.body))])
		}
	}
}

func TestModelReadinessChecksState(t *testing.T) {
	for _, state := range []string{"AVAILABLE", "LOADING"} {
		backend := &modelClient{url: "http://127.0.0.1/model", client: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.Method != "GET" {
				t.Fatal("readiness must not perform inference")
			}
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"model_version_status":[{"version":"2","state":"` + state + `","status":{"error_code":"OK"}}]}`))}, nil
		})}}
		err := backend.Ready(context.Background())
		if (err == nil) != (state == "AVAILABLE") {
			t.Errorf("state=%s err=%v", state, err)
		}
	}
}
