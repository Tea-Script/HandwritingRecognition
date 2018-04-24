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
    for i in range(len(latex)):
        try:
            c = latex[i]
            if(c == "\\"): #found a "\" command
                sym = found_symbols.pop(0)
                arr.append(sym)
                latex = latex.replace(sym, "", 1)

            elif(c in "{}^_ " ): #found a position delimeter or container that doesn't belong to a command
                pass
            else: #found a normal character
                arr.append(c)
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
        self.M = np.array([[-1]*(len(b))]*(len(a)))
        return self.levenhelper(a, b, len(a), len(b) )

    def levenhelper(self, a, b, i, j):
        '''for recursive levenshtein distance dynamic programming
        ***Gives the distance between the first i characters of a and the first j characters of b.
        '''
        if(i == 0 or j == 0 or j > len(b) or i > len(a)):
                return max(i,j)
        elif(self.M[i,j] != -1):
            return self.M[i,j]
        else:
            self.M[i,j] = min([
                self.levenhelper(a, b, i - 1, j) + 1,
                self.levenhelper(a, b, i, j + 1) + 1,
                self.levenhelper(a, b, i-1, j-1) + (a[i - 1] != b[j - 1])
            ])
            return self.M[i,j]

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

