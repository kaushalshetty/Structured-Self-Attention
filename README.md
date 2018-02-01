# Structured Self-attentive sentence embeddings 
---
#### USAGE:
For binary sentiment classification on imdb dataset run :
`python classification.py "binary"`

For multiclass classification on reuters dataset run :
`python classification.py "multiclass"`

You can change the model parameters in the model_params.json file
Other tranining parameters like number of attention hops etc can be configured in the config.json file.

If you want to use pretrained glove embeddings , set the `use_embeddings` parameter to `"True"` ,default is set to False. Do not forget to download the `glove.6B.50d.txt` and place it in the glove folder.

---

#### Implemented:
* Classification using self attention
* Regularization using Frobenius norm
* Gradient clipping
* Visualizing the attention weights

Instead of pruning ,used averaging over the sentence embeddings.

#### Visualization:
After training, the model is tested on 100 test points. Attention weights for the 100 test data are retrieved and used to visualize over the text using heatmaps. The visualization code was provided by Zhouhan Lin. Many thanks.


Below is a shot of the visualization on few datapoints.
![alt text](https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/visualization/attention.png "Attention Visualization")

