# Hamburger image-classification demo

A Go upload page and API for a small TensorFlow model trained around ~2018 to distinguish
hamburgers from other images. The original training script and training images are
lost; preserve the checked-in SavedModel, variables, and labels before migration.
The public demo uses `/hamppa/` and `POST /api/v1/inference` on laurikos.net.

### UPDATE @ 2024-08-06

Hosted _currently_ at `https://laurikos.net/hamppa/`
This is quite old project where I was doing demo of how to serve a tensorflow model trhough API

Main point of the demo was to show basically how to make the saved_model as protobuf with indexes and labels
and serve that via HTTP API.
I could not find the original training.py script, nor the images used to train it.
In short, tt was following the tensorflow example mentioned before.
I think I downloaded some 100 pictures of hamburgers from _internet_
And I remeber also using some public image library for training NNs with some random everyday objects,
from where I also gathered some 100 pictures.

---

**This demo project was inspired by HBOs "Silicon Valley" tv-series**
where they developed a "SeeFood" application, if memory serves, that was capable of classifying
if image was hotdog or not.
So obviously I then created the same for hamburgers.

I think this project was done in around 2018 - so not that ancient but
it was nice to notice that old Go project with some dependencies (and quite horrible dockerimage) still managed
to work without any modifications to it.

### UPDATE @ 2026-11-01

I let AI update some dependencies and do some cleanup

---

## Runtime

The HTTP application uses the Go standard library. It decodes JPEG/PNG, resizes to
299 × 299 RGB with the original TensorFlow `ResizeBilinear` coordinate convention,
normalizes by 255, and calls TensorFlow Serving on loopback. JPEG decoders can differ
slightly; this is not a claim of bit-for-bit parity with the old native decoder.
The trained model is unchanged. Its `serving_default` signature accepts float32
`image` shaped `[batch, 299, 299, 3]` and returns `prediction` shaped `[batch, 2]`.
Labels remain `hamburger`, then `not hamburger`.

The former TensorFlow 1.12 cgo bindings and per-request native sessions are no longer
part of the Go application. The prepared model service uses the official
`tensorflow/serving:2.21.0` Linux/amd64 image, pinned to this registry digest:

```text
sha256:b09456481317351201fbe0b3c5954273e1b697b4a4a7834dc56f01731137a529
```

The image's native `tensorflow_model_server` binary is invoked directly. Its model
mount is read-only, it runs as UID/GID 65534, and only the REST port is published,
on `127.0.0.1:8501`. gRPC is not published. The app binds to `127.0.0.1:9999`.

The unchanged model was verified on Linux/amd64 with this runtime on 2026-11-01:
JPEG/PNG inference and 60 repeated requests passed, with a 370 MiB model memory
peak and no OOM events. These synthetic fixtures verify execution, not classification
accuracy or indefinite-load behavior. See the deployment notes for the full limits.

## Build and test

Use the already installed Go 1.27.3 or newer. No Go modules, Python packages, native
TensorFlow headers, or JavaScript packages are needed to build the HTTP application.

```sh
make test
go test -race ./...
make build
make linux
```

The Linux target is a static amd64 binary at `bin/hamppa-linux-amd64`. With an approved
TensorFlow Serving installation running locally:

```sh
./bin/hamppa --addr=127.0.0.1:9999
curl --fail http://127.0.0.1:9999/readyz
curl --fail -F image=@example.png http://127.0.0.1:9999/api/v1/inference
```

Flags can override `--model-url`, `--labels`, and `--index`. The model URL must use
`http://127.0.0.1`; proxy environment variables and HTTP redirects are disabled for
model requests. `/healthz` checks the HTTP process; `/readyz` checks model readiness.

## Upload limits and failure behavior

- One request at a time, including body parsing and image normalization. Concurrent
  submissions return 429 with `Retry-After: 1`.
- Requests up to 10 MiB; one image up to 8 MiB, 16 million pixels, and 8192 pixels per
  dimension. Dimensions are checked before decoding the complete image.
- Image contents determine JPEG/PNG format, regardless of filename or extension.
- Malformed input returns 400, size-limit failures return 413, and model errors
  return 502 without stopping the HTTP process.
- Model requests have a 30-second client limit; TensorFlow Serving is configured
  for a 20-second REST execution timeout and one inter/intra-op worker each.
- The existing `filename` and sorted `labels` JSON response is preserved.

Tests cover valid formats, normalization coordinates, transparent PNG RGB values,
malformed/oversized uploads, a dimension bomb, concurrent requests, backend failures,
invalid outputs, model request format/readiness, routing, and listen errors.
They mock the model server and require no external requests.

## Linux deployment

See [deployment files and validation](deploy/README.md). The systemd files are
prepared for `/opt/hamppa`. They must be reviewed before installing server software.
The Dockerfile is an optional scratch image for the prebuilt HTTP application;
normal deployment runs that application directly under systemd. Building that image
requires an explicit approved Docker build and a Linux binary produced by `make linux`.
If used on Linux, it needs host networking to reach the loopback model service.

## Historical context

The model was (re)trained with roughly 100 hamburger images and 100 miscellaneous
images following TensorFlow's image-retraining example. This was a learning demo
inspired by the SeeFood application in HBO's Silicon Valley, not a validated
classifier. The previous 2024 deployment used a TensorFlow 1.12/Go 1.11 container;
its model files are the source of truth for migration until compared with this copy.
