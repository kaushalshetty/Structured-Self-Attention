# Structured Self-attentive sentence embeddings 
Implementation for the paper A Structured Self-Attentive Sentence Embedding, which was published in ICLR 2017: https://arxiv.org/abs/1703.03130 .
#### USAGE:
For binary sentiment classification on imdb dataset run :
`python classification.py "binary"`

For multiclass classification on reuters dataset run :
`python classification.py "multiclass"`

You can change the model parameters in the `model_params.json file`
Other tranining parameters like number of attention hops etc can be configured in the `config.json` file.

If you want to use pretrained glove embeddings , set the `use_embeddings` parameter to `"True"` ,default is set to False. Do not forget to download the `glove.6B.50d.txt` and place it in the glove folder.



#### Implemented:
* Classification using self attention
* Regularization using Frobenius norm
* Gradient clipping
* Visualizing the attention weights

Instead of pruning ,used averaging over the sentence embeddings.

#### Visualization:
After training, the model is tested on 100 test points. Attention weights for the 100 test data are retrieved and used to visualize over the text using heatmaps. A file visualization.html gets saved in the visualization/ folder after successful training. The visualization code was provided by Zhouhan Lin (@hantek). Many thanks.


Below is a shot of the visualization on few datapoints.
![alt text](https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/visualization/attention.png "Attention Visualization")



Training accuracy 93.4%
Tested on 1000 points with 90.2% accuracy

---

