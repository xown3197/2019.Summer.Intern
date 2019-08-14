import numpy as np
from scipy.cluster.vq import vq
import pickle

def make_hist(codeword, bins = 200):
   hist, _ = np.histogram(codeword, bins=bins)
   return hist

def make_codeword(features, codebook):
   codeword, _ = vq(features, codebook)
   return codeword

def load_codebook(path):
    codebook = pickle.load(open('{}'.format(path), 'rb'))
    return codebook