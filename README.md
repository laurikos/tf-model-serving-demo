## Demo for serving tensorflow model via API endpoint
Quickly made GO server with API endpoint to handle image classification
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
> -> run docker image at :8888
>
> -> optionally deploy to kubernetes
>
> ->
> POST /api/v1/inference with form data "image" : "imagename.jpg"


### Todo:
Pretty much everything; this is quickly made demo