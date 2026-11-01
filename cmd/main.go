package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

func run(ctx context.Context, server *http.Server) error {
	result := make(chan error, 1)
	go func() { result <- server.ListenAndServe() }()
	select {
	case err := <-result:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		stopCtx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
		defer cancel()
		if err := server.Shutdown(stopCtx); err != nil {
			_ = server.Close()
			return err
		}
		if err := <-result; !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	}
}

func loadLabels(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	labels := strings.FieldsFunc(string(data), func(r rune) bool { return r == '\n' || r == '\r' })
	if len(labels) != 2 || labels[0] == labels[1] {
		return nil, fmt.Errorf("expected two distinct model labels")
	}
	return labels, nil
}

func main() {
	addr := flag.String("addr", "127.0.0.1:9999", "HTTP listen address")
	modelURL := flag.String("model-url", "http://127.0.0.1:8501/v1/models/hamppa/versions/2", "TensorFlow Serving version URL")
	labelsPath := flag.String("labels", "models/labels/hamppa/saved_model_labels.txt", "model labels")
	indexPath := flag.String("index", "index.html", "upload page")
	flag.Parse()
	endpoint, err := url.Parse(*modelURL)
	if err != nil || endpoint.Scheme != "http" || endpoint.Hostname() != "127.0.0.1" || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.Fragment != "" {
		log.Fatal("model-url must be an HTTP URL on 127.0.0.1 without credentials, query, or fragment")
	}
	labels, err := loadLabels(*labelsPath)
	if err != nil {
		log.Fatal(err)
	}
	index, err := os.ReadFile(*indexPath)
	if err != nil {
		log.Fatal(err)
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.MaxConnsPerHost = 2
	backend := &modelClient{
		url: strings.TrimRight(endpoint.String(), "/"),
		client: &http.Client{Transport: transport, Timeout: 30 * time.Second,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
	}
	app := newApplication(backend, labels, index)
	server := &http.Server{
		Addr: *addr, Handler: app, ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 15 * time.Second, WriteTimeout: 45 * time.Second,
		IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16 << 10,
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Printf("Serving hamburger demo on %s", *addr)
	if err := run(ctx, server); err != nil {
		log.Fatal(err)
	}
	log.Print("HTTP server stopped")
}
