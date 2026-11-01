# Optional image for the HTTP application. Build the static binary with make linux.
# The trained model runs separately in the pinned TensorFlow Serving image.
FROM scratch
WORKDIR /app
COPY bin/hamppa-linux-amd64 /app/hamppa
COPY index.html /app/index.html
COPY models/labels/hamppa/saved_model_labels.txt /app/models/labels/hamppa/saved_model_labels.txt
USER 65534:65534
ENTRYPOINT ["/app/hamppa"]
