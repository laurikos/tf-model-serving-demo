package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

type modelClient struct {
	url    string
	client *http.Client
}

func (m *modelClient) Predict(ctx context.Context, image imageTensor) ([]float32, error) {
	request := struct {
		Signature string                   `json:"signature_name"`
		Inputs    map[string][]imageTensor `json:"inputs"`
	}{"serving_default", map[string][]imageTensor{"image": {image}}}
	body, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.url+":predict", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	var response struct {
		Outputs json.RawMessage `json:"outputs"`
	}
	if err := m.request(req, &response); err != nil {
		return nil, err
	}
	var output [][]float32
	if err := json.Unmarshal(response.Outputs, &output); err != nil {
		var named struct {
			Prediction [][]float32 `json:"prediction"`
		}
		if err := json.Unmarshal(response.Outputs, &named); err != nil {
			return nil, errors.New("invalid prediction response")
		}
		output = named.Prediction
	}
	if len(output) != 1 {
		return nil, errors.New("expected one prediction")
	}
	return output[0], nil
}

func (m *modelClient) Ready(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, m.url, nil)
	if err != nil {
		return err
	}
	var response struct {
		Versions []struct {
			Version string `json:"version"`
			State   string `json:"state"`
			Status  struct {
				ErrorCode string `json:"error_code"`
			} `json:"status"`
		} `json:"model_version_status"`
	}
	if err := m.request(req, &response); err != nil {
		return err
	}
	for _, version := range response.Versions {
		if version.State == "AVAILABLE" && version.Status.ErrorCode == "OK" {
			return nil
		}
	}
	return errors.New("model is not ready")
}

func (m *modelClient) request(req *http.Request, out any) error {
	resp, err := m.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("model server HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, (64<<10)+1))
	if err != nil {
		return err
	}
	if len(body) > 64<<10 {
		return errors.New("model response too large")
	}
	return json.Unmarshal(body, out)
}
