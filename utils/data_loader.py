#Please create your own dataloader for new datasets of the following type

import torch
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import torch.utils.data as data_utils
 
def load_data_set(type,max_len,vocab_size,batch_size):
    """
        Loads the dataset. Keras Imdb dataset for binary classifcation. Keras reuters dataset for multiclass classification
 
        Args:
            type   : {bool} 0 for binary classification returns imdb dataset. 1 for multiclass classfication return reuters set
            max_len: {int} timesteps used for padding
			vocab_size: {int} size of the vocabulary
			batch_size: batch_size
        Returns:
            train_loader: {torch.Dataloader} train dataloader
            x_test_pad  : padded tokenized test_data for cross validating
			y_test      : y_test
            word_to_id  : {dict} words mapped to indices
 
      
        """
   
    INDEX_FROM=3
    if not bool(type):
        NUM_WORDS=vocab_size # only use top 1000 words
           # word index offset
 
        train_set,test_set = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
        x_train,y_train = train_set[0],train_set[1]
        x_test,y_test = test_set[0],test_set[1]
        word_to_id = imdb.get_word_index()
        word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
 
        id_to_word = {value:key for key,value in word_to_id.items()}
        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])
        n_train = x.shape[0] - 1000
        n_valid = 1000
 
        x_train = x[:n_train]
        y_train = y[:n_train]
        x_test = x[n_train:n_train+n_valid]
        y_test = y[n_train:n_train+n_valid]
 
 
        #embeddings = load_glove_embeddings("../../GloVe/glove.6B.50d.txt",word_to_id,50)
        x_train_pad = pad_sequences(x_train,maxlen=max_len)
        x_test_pad = pad_sequences(x_test,maxlen=max_len)
 
 
        train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.DoubleTensor))
        train_loader = data_utils.DataLoader(train_data,batch_size=batch_size,drop_last=True)
        return train_loader,x_test_pad,y_test,word_to_id
       
    else:
        from keras.datasets import reuters
 
        train_set,test_set = reuters.load_data(path="reuters.npz",num_words=vocab_size,skip_top=0,index_from=INDEX_FROM)
        x_train,y_train = train_set[0],train_set[1]
        x_test,y_test = test_set[0],test_set[1]
        word_to_id = reuters.get_word_index(path="reuters_word_index.json")
        word_to_id = {k:(v+3) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        word_to_id['<EOS>'] = 3
        id_to_word = {value:key for key,value in word_to_id.items()}
        x_train_pad = pad_sequences(x_train,maxlen=max_len)
        x_test_pad = pad_sequences(x_test,maxlen=max_len)
 
 
        train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.LongTensor))
        train_loader = data_utils.DataLoader(train_data,batch_size=batch_size,drop_last=True)
        return train_loader,train_set,test_set,x_test_pad,word_to_id