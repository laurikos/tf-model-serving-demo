package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/go-chi/chi"
	"github.com/go-chi/chi/middleware"
	"github.com/go-chi/cors"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	savedModel   *tf.SavedModel
	graphModel   *tf.Graph
	sessionModel *tf.Session
	labels       []string
)

// ClassifyResult JSON type result for labelname
type ClassifyResult struct {
	Filename string        `json:"filename"`
	Labels   []LabelResult `json:"labels"`
}

// LabelResult JSON type result for label and prob
type LabelResult struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}

func main() {
	// Load tensorflow model:
	workdir, _ := os.Getwd()
	log.Println("workDir: " + workdir)

	// Load model, use loadModel() function if using just graph file instead of saved model
	model, err := tf.LoadSavedModel(workdir+"/models/myfirstmodel/2", []string{"serve"}, nil)
	if err != nil {
		panic(err)
	}
	savedModel = model

	// Load labels
	labelsFile, err := os.Open(workdir + "/models/labels/hamppa/saved_model_labels.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)
	// Labels are separated by newlines
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	// Start server with chi-router with middleware and cors
	router := chi.NewRouter()

	cors := cors.New(cors.Options{
		// AllowedOrigins: []string{"https://foo.com"}, // Use this to allow specific origin hosts
		AllowedOrigins: []string{"*"},
		// AllowOriginFunc:  func(r *http.Request, origin string) bool { return true },
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:           300, // Maximum value not ignored by any of major browsers
	})

	router.Use(cors.Handler)

	router.Use(middleware.RequestID)
	router.Use(middleware.RealIP)
	router.Use(middleware.Logger)
	router.Use(middleware.Recoverer)

	router.Use(middleware.Timeout(60 * time.Second))

	router.Post("/api/v1/inference", inferenceHandler)

	// Serve the index.html file at the root path
	router.Get("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, filepath.Join(workdir, "index.html"))
	})


	log.Fatal(http.ListenAndServe(":8888", router))

}

func inferenceHandler(w http.ResponseWriter, r *http.Request) {
	// Read image
	imageFile, header, err := r.FormFile("image")
	// Will contain filename and extension
	imageName := strings.Split(header.Filename, ".")
	if err != nil {
		responseError(w, "Could not read image", http.StatusBadRequest)
		return
	}
	defer imageFile.Close()
	var imageBuffer bytes.Buffer
	// Copy image data to a buffer
	io.Copy(&imageBuffer, imageFile)

	log.Println("input image name: " + imageName[:1][0])
	// ...
	// Make tensor
	tensor, err := makeTensorFromImage2(&imageBuffer, imageName[:1][0])
	if err != nil {
		responseError(w, "Invalid image", http.StatusBadRequest)
		return
	}

	log.Printf("image to tensor shape %d: \n", tensor.Shape())

	feedsOutput := tf.Output{
		Op:    savedModel.Graph.Operation("Placeholder"),
		Index: 0,
	}
	feeds := map[tf.Output]*tf.Tensor{feedsOutput: tensor}

	fetches := []tf.Output{
		{
			Op:    savedModel.Graph.Operation("final_result"),
			Index: 0,
		},
	}

	output, err := savedModel.Session.Run(feeds, fetches, nil)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("output: [ %g, %g ]\n", (output[0].Value().([][]float32)[0][0]), (output[0].Value().([][]float32)[0][1]))

	// Return best labels
	responseJSON(w, ClassifyResult{
		Filename: header.Filename,
		Labels:   findBestLabels(output[0].Value().([][]float32)[0]),
	})

}

func readImage(url string) []byte {
	file, err := os.Open(url)
	if err != nil {
		panic(err)
	}

	defer file.Close()

	b, err := ioutil.ReadAll(file)
	if err != nil {
		panic(err)
	}

	return b
}

func responseError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func responseJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

// ByProbability Sorts result and labels by result
type ByProbability []LabelResult

func (a ByProbability) Len() int           { return len(a) }
func (a ByProbability) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByProbability) Less(i, j int) bool { return a[i].Probability > a[j].Probability }

func findBestLabels(probabilities []float32) []LabelResult {
	// Make a list of label/probability pairs
	var resultLabels []LabelResult
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		resultLabels = append(resultLabels, LabelResult{Label: labels[i], Probability: p})
	}
	// Sort by probability
	sort.Sort(ByProbability(resultLabels))
	// Return top 2 labels
	return resultLabels[:2]
}

// Used with imported tensorflow graph instead of saved model
func loadModels() {
	workdir, _ := os.Getwd()
	modelfile, err := ioutil.ReadFile(workdir + "/models/graph/test/saved_model.pb")
	if err != nil {
		log.Fatal(err)
	}

	graphModel = tf.NewGraph()
	if err := graphModel.Import(modelfile, ""); err != nil {
		log.Fatal(err)
	}

	sessionModel, err = tf.NewSession(graphModel, nil)
	if err != nil {
		log.Fatal(err)
	}
}
