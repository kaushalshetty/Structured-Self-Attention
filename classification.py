#You can write your own classification file to use the module
from attention.model import StructuredSelfAttention
from attention.train import train,get_activation_wts,evaluate
from utils.pretrained_glove_embeddings import load_glove_embeddings
from utils.data_loader import load_data_set
from visualization.attention_visualization import createHTML
import torch
import numpy as np
from torch.autograd import Variable
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import torch.utils.data as data_utils
import os,sys
import json
 
classified = False
classification_type = sys.argv[1]
 
def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == 'False':
            json_set[k] = False
        elif v == 'True':
            json_set[k] = True
        else:
            json_set[k] = v
    return json_set
 
 
with open('config.json', 'r') as f:
    params_set = json.load(f)
 
with open('model_params.json', 'r') as f:
    model_params = json.load(f)
 
params_set = json_to_dict(params_set)
model_params = json_to_dict(model_params)
 
print("Using settings:",params_set)
print("Using model settings",model_params)
 
def visualize_attention(wts,x_test_pad,word_to_id,filename):
    wts_add = torch.sum(wts,1)
    wts_add_np = wts_add.data.numpy()
    wts_add_list = wts_add_np.tolist()
    id_to_word = {v:k for k,v in word_to_id.items()}
    text= []
    for test in x_test_pad:
        text.append(" ".join([id_to_word.get(i) for i in test]))
    createHTML(text, wts_add_list, filename)
    print("Attention visualization created for {} samples".format(len(x_test_pad)))
    return
 
def binary_classfication(attention_model,train_loader,epochs=5,use_regularization=True,C=1.0,clip=True):
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(attention_model.parameters())
    train(attention_model,train_loader,loss,optimizer,epochs,use_regularization,C,clip)
 
def multiclass_classification(attention_model,train_loader,epochs=5,use_regularization=True,C=1.0,clip=True):
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.RMSprop(attention_model.parameters())
    train(attention_model,train_loader,loss,optimizer,epochs,use_regularization,C,clip)
 
 
 
MAXLENGTH = model_params['timesteps']
if classification_type =='binary':
 
    train_loader,x_test_pad,y_test,word_to_id = load_data_set(0,MAXLENGTH,model_params["vocab_size"],model_params['batch_size']) #loading imdb dataset
 
 
    if params_set["use_embeddings"]:
        embeddings = load_glove_embeddings("glove/glove.6B.50d.txt",word_to_id,50)
    else:
        embeddings = None
    #Can use pretrained embeddings by passing in the embeddings and setting the use_pretrained_embeddings=True
    attention_model = StructuredSelfAttention(batch_size=train_loader.batch_size,lstm_hid_dim=model_params['lstm_hidden_dimension'],d_a = model_params["d_a"],r=params_set["attention_hops"],vocab_size=len(word_to_id),max_len=MAXLENGTH,type=0,n_classes=1,use_pretrained_embeddings=params_set["use_embeddings"],embeddings=embeddings)
 
    #Can set use_regularization=True for penalization and clip=True for gradient clipping
    binary_classfication(attention_model,train_loader=train_loader,epochs=params_set["epochs"],use_regularization=params_set["use_regularization"],C=params_set["C"],clip=params_set["clip"])
    classified = True
    #wts = get_activation_wts(binary_attention_model,Variable(torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
    #print("Attention weights for the testing data in binary classification are:",wts)
 
 
if classification_type == 'multiclass':
    train_loader,train_set,test_set,x_test_pad,word_to_id = load_data_set(1,MAXLENGTH,model_params["vocab_size"],model_params['batch_size']) #load the reuters dataset
    #Using pretrained embeddings
    if params_set["use_embeddings"]:
        embeddings = load_glove_embeddings("glove/glove.6B.50d.txt",word_to_id,50)
    else:
        embeddings = None
    attention_model = StructuredSelfAttention(batch_size=train_loader.batch_size,lstm_hid_dim=model_params['lstm_hidden_dimension'],d_a = model_params["d_a"],r=params_set["attention_hops"],vocab_size=len(word_to_id),max_len=MAXLENGTH,type=1,n_classes=46,use_pretrained_embeddings=params_set["use_embeddings"],embeddings=embeddings)
 
    #Using regularization and gradient clipping at 0.5 (currently unparameterized)
    multiclass_classification(attention_model,train_loader,epochs=params_set["epochs"],use_regularization=params_set["use_regularization"],C=params_set["C"],clip=params_set["clip"])
    classified=True
    #wts = get_activation_wts(multiclass_attention_model,Variable(torch.from_numpy(x_test_pad[:]).type(torch.LongTensor)))
    #print("Attention weights for the data in multiclass classification are:",wts)
if classified:
    test_last_idx = 100
    wts = get_activation_wts(attention_model,Variable(torch.from_numpy(x_test_pad[:test_last_idx]).type(torch.LongTensor)))
    print(wts.size())
    visualize_attention(wts,x_test_pad[:test_last_idx],word_to_id,filename='attention.html')