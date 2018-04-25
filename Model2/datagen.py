import subprocess
from scipy.misc import *
import random
import subprocess
import sys
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage import io
random.seed(77)

with open('latexsymbols.txt', 'r') as f:
    supported_characters = f.read().split('\n')

    
TRAINING_SIZE = 50
TEST_SIZE = 25
SIZES = [TRAINING_SIZE, TEST_SIZE]


def make_img(symbol, phase, n, plot = False):
    '''Creates an image of a symbol'''
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
    if(sym == '' or sym==" "):
        return




    sym = sym.replace("@", "at") #replaced invalid directory symbols
    path = os.path.join(phase, sym)
    if(not os.path.isdir(path)):
        os.system("mkdir " + path)
    print(sym)
    
    latexheader = """
    \\documentclass[12pt]{article}
    \\pagenumbering{gobble}
    \\usepackage{amsmath}
    \\usepackage{amssymb}
    \\usepackage{euler}
    \\usepackage{concrete}
    \\begin{document}
    \\Huge{$
    """
    latexend = """
    $}\\end{document}
    """
    #generate latex documents

    latex_doc = latexheader + symbol + latexend
    f= open("{}.tex".format(path),"w+")
    f.write(latex_doc)
    f.close()
    #compile latex documents
    #dependency: texlive; bash
    subprocess.check_output(['pdflatex', '-output-directory='+phase, '{}.tex'.format(path)]) #make pdf in train, from image
    #convert pdfs to images jpg
    #dependency: ImageMagick; bash
    imgloc = '{}/{}.jpg'.format(path, sym)
    subprocess.check_output(['convert', '-quality', '100',  '{}.pdf'.format(path), imgloc])
    #now we need to read the images in as arrays and segment them into 56x56px arrays consisting of all the black squares.
    #dependency: Scipy
    image = io.imread(imgloc, mode="L")/255
    #for each array in arrays, to center an image of a digit, we should identify the width and height of the digit
    
    cond = image < .2 #threshold for dark enough
    indices = np.argwhere(cond)
    minarr = np.amin(indices, axis = 0) #get first row, col meeting cond
    maxarr = np.amax(indices, axis = 0) #get last row, col meeting cond
    tmp = image[minarr[0]:maxarr[0] + 1, minarr[1] : maxarr[1] + 1]
        
    hw = maxarr - minarr + 1 #height, width
        
    #if(np.any(hw > (32,32))): raise Exception("Image too large, digit requires compression") #this will never happen with our set
    #we need to center the digit in the 32x32 space
    remainder_height = (32 - hw[0]) // 2 
    remainder_width = (32 - hw[1]) // 2
    arr = np.ones((32,32))
    #tmp = image[minarr[0] : minarr, startcol : fincol]
    arr[remainder_height : hw[0] + remainder_height , remainder_width : hw[1] + remainder_width] = tmp  
    if(plot):
        plt.imshow(arr, cmap=plt.get_cmap('Greys_r'))
        plt.show()
    os.system("rm " + imgloc)
    for i in range(n):
        plt.imsave(imgloc[:-4] +str(i)+".png" , tmp, format="png", cmap=plt.get_cmap('Greys_r'))
    return arr



#r = list(range(33, 123)) #keyboard values of ascii table
blacklist = [91,92,93,94,95,35,36,37,38, 39]
r = [33, 43,45 ] + list(range(48,57)) + [60, 61, 62] + list(range(65,91)) + list(range(97,123))
r = [chr(x) for x in r if x not in blacklist] #remove special characters and escape characters
class_names = r + supported_characters 

def generate_samples(n, phase, plot=0):
    '''Creates n samples and puts them in their respective folder'''
    for symbol in class_names:
        make_img(symbol, phase, n, plot=plot)
def clear_samples(phase):
    path = os.path.join(phase, "*")
    try:
        os.system("rm -r " + path)
        os.system("rm " + path)
    except:
        pass

if(__name__ == "__main__"):
    phases = ["train", "test"]
    [clear_samples(x)  for x in phases]
    [generate_samples(SIZES[i], x)for i,x in enumerate(phases)]
    print("Done!")





    
