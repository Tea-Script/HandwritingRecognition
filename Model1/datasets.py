import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import *
from skimage import io, transform
import scipy.ndimage as sci
import re
plt.ion()

with open('latexsymbols.txt', 'r') as f:
    supported_characters = f.read().split('\n')


data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomRotation(75),
        #transforms.RandomAffine(130, translate=(15,15)),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])


        
data_dir = "./"

def simplify(path, verbose=0):
    '''Returns a list containing each character in a LaTex document string in order of appearance'''
    with open(path, "r+") as f:
    	latex = f.read().replace('\n',"") 
    start = latex.find("\\begin{minipage}[t][0pt]{\\linewidth}") + len("\\begin{minipage}[t][0pt]{\\linewidth}")
    end = latex.find("\\end{minipage}")
    latex = latex[start: end] #document body
    latex = latex.replace("\\[","") #removing noncharacter command for equations
    latex = latex.replace("\\]","") #removing noncharacter command for equations
    #latex = latex.replace("$", "") #removing noncharacter command for equations
    #latex = latex.replace("\text{") should make a function for this if necessary later on
    #find all supported LaTeX commands
    escaped_chars = [re.escape(x) for x in supported_characters]
    found_symbols = re.findall(r"(?=("+'|'.join(escaped_chars)+r"))", latex)
    found_symbols = list(filter(None, found_symbols)) #remove empty strings 
    #search latex file. For every "\" add the next found supported word to a list, then remove its first occurence from the string and list
    arr = []
    if(verbose): print(found_symbols)
    i = 0
    while i < len(latex):
        try:
            c = latex[i]
            if(c == "\\"): #found a "\" command
                sym = found_symbols.pop(0)
                arr.append(sym)
                i += len(sym) - 1

            elif(c in "{}^_ " ): #found a position delimeter or container that doesn't belong to a command
                pass
            else: #found a normal character
                arr.append(c)
            i += 1
        except IndexError:
            if(verbose): print("Warning no symbols remaining")
            break
    return arr
class lev:
    def __init__(self):
        pass
    def Levenshtein(self, observed, expected, simplify = False):
        '''Determines the Levenshtein distance between the ordered lists; Uses Memoization'''
        if(simplify):
            a = simplify(observed)
            b = simplify(expected)
        else:
            a = observed
            b = expected
        if(len(a) > len(b)):
            self.M = np.ones((len(a), len(b)))* -1
            return self.levenhelper(a, b, len(a), len(b) )
        else:
            self.M = np.ones((len(b), len(a)))* -1
            return self.levenhelper(b, a, len(b), len(a) )

    def levenhelper(self, a, b, i, j):
        '''for recursive levenshtein distance dynamic programming
        ***Gives the distance between the first i characters of a and the first j characters of b.
        '''
        if(min(i,j) == 0 or i > len(a) or j > len(b) ):
            return max(i,j)
        elif(self.M[i - 1, j - 1] != -1):
            return self.M[i - 1, j - 1]
        else:
            l1 = self.levenhelper(a, b, i - 1, j) + 1 
            l2 = self.levenhelper(a, b, i, j + 1) + 1
            l3 = self.levenhelper(a, b, i-1, j-1) + (a[i - 1] != b[j - 1])
            self.M[i - 1, j - 1] = min([l1,l2,l3])
            return self.M[i - 1, j - 1]

def replace(symbol):
    '''Removes characters not allowable by linux directory names and replaces them with usable names'''
    sym = symbol.replace("\\", "")
    sym = sym.replace("'", "squote")
    sym = sym.replace('"', "dquote")
    sym = sym.replace("(", "leftpar")
    sym = sym.replace(")", "rightpar")
    sym = sym.replace("*", "star")
    sym = sym.replace(",", "comma")
    sym = sym.replace(".", "period")
    sym = sym.replace(";", "semicolon")
    sym = sym.replace(":", "colon")
    sym = sym.replace("?", "question")
    sym = sym.replace("!", "bang")
    sym = sym.replace("~", "tilde")
    sym = sym.replace("+", "plus")
    sym = sym.replace("=", "equals")
    sym = sym.replace("-", "minus")
    sym = sym.replace("/", "root")
    sym = sym.replace("<", "less")
    sym = sym.replace(">", "great")
    sym = sym.replace("`", "backtick")
    sym = sym.replace("&", "and")
    return sym



LEV = lev()


if(__name__ == "__main__"):
    print(simplify("test/0.tex", verbose=1))
    print(2, LEV.Levenshtein(["123", "222"], ["222","123"]))
    print(1,LEV.Levenshtein(["111"],["112"]))
    print(0, LEV.Levenshtein(["111"],["111"]))
    print(2, LEV.Levenshtein(["111"],["123","122"])) 
    print(1, LEV.Levenshtein(["131","111","121"],["131","151", "111","121"]))
    print(3, LEV.Levenshtein(["131", "121"],["131","144", "111", "121","141"]))
    

