## Demo for serving tensorflow model via API endpoint

Quickly made Go server with API endpoint to handle image classification
Based on tensorflow example
https://github.com/tensorflow/hub/tree/master/examples/image_retraining

Works with retrain.py's outputs, either saved_model or outputted-graph

## Usage:

> git clone ->
>
> optionally use retrain.py to create your own model and change saved_model_dir location
> at tf.loadSavedModel()
>
> -> build docker image
>
> -> run docker image with -p HOSTPORT:8888
>
> ->
> POST /api/v1/inference with form data "image" : "imagename.jpg"

### Todo:

Pretty much everything; this is quickly made demo

---

# UPDATE @ 2024-08-06

Hosted _currently_ at `https://laurikos.net/hamppa/`

This is quite old project where I was doing demo of how to serve a tensorflow model trhough API

Main point of the demo was to show basically how to make the saved_model as protobuf with indexes and labels
and serve that via HTTP API.

I could not find the original training.py script, nor the images used to train it.

In short, tt was following the tensorflow example mentioned before.
I think I downloaded some 100 pictures of hamburgers from _internet_
And I remeber also using some public image library for training NNs with some random everyday objects, from where I also gathered some 100 pictures.

---

This demo project was inspired by HBOs "Silicon Valley" tv-series where they developed a "SeeFood" application, if memory serves, that was
capable of classifying if image was hotdog or not. So obviously I then created the same for hamburgers.

---

I think this project was done in around 2018 - so not that ancient but
it was nice to notice that old Go project with some dependencies (and quite horrible dockerimage) still managed
to work without any modifications to it.

---
