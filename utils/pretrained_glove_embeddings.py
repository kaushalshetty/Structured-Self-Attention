from nltk import word_tokenize
import numpy as np
import torch
 
vocab,word2idx = None,{}
 
 
def load_glove_embeddings(path, word2idx, embedding_dim):
    """Loading the glove embeddings"""
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                if vector.shape[-1] != embedding_dim:
                    raise Exception('Dimension not matching.')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()
def get_vocab():
    """
    Get vocabulary. Must run get_embeddings first
   
    Returns:
        vocabulary
   
    """
    if vocab is None:
        raise Exception("Run get_embedding first to create vocabulary")
    return vocab
def get_word_idx():
    """
    Get word2idx. Must run get_embeddings first
   
    Returns:
        word2idx
   
    """
    if not word2idx:
        raise Exception("Run get_embedding first to create vocabulary")
    return word2idx
 
 
 
def get_embeddings(emb_path,corpus_tokens,emb_dim,add_eos=False,add_sos=False,add_unk=False,add_pad=False):
    """
        Method to get the embeddings
 
        Args:
            emb_path     : {path} path of glove embeddings
            corpus_tokens: {list} list of tokens from the corpus. Use keras or spaCy tokenizer to tokenize
            emb_dim      : {int} embeddings dimension
            add_eos      : {bool} add <EOS> tag to vocab
            add_sos      : {bool} add <SOS> tag to vocab
            add_unk      : {bool} add <UNK> tag to vocab
            add_pad      : {bool} add <PAD> tag to vocab
 
        Returns:
            gloVe Embeddings
 
        
    """
    global vocab,word2idx
    vocab = set(corpus_tokens)
    addons = int(add_eos)+int(add_unk)+int(add_sos)+int(add_pad)
    word2idx = {word: (idx+addons) for idx, word in enumerate(vocab)}
   
    if add_pad:
        word2idx['<PAD>'] = 0
    if add_eos:
        word2idx['<EOS>'] = 1
    if add_sos:
        word2idx['<SOS>'] = 2
    if add_unk:
        word2idx['<UNK>'] = 3
   
    
    #print(word2idx)
    # create word index
    word_embeddings = load_glove_embeddings(emb_path, word2idx,emb_dim)
    word_embedding = torch.nn.Embedding(word_embeddings.size(0), word_embeddings.size(1))
    word_embedding.weight = torch.nn.Parameter(word_embeddings)
    return word_embedding