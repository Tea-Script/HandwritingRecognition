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

    
TRAINING_SIZE = 25
TEST_SIZE = 15
SIZES = [TRAINING_SIZE, TEST_SIZE]

fonts = ["default", "euler", "roman", "typewriter", "serif", "bb", "Antywaka condensed",
             "mathit", "mathpazo", "boisik", "sanscomputer", "neohellenic" , "bakersville", 
             "custom", "Antykwa"
            ]
fonts = ["euler"]


def gen_latex(digit, font="default"):
    '''Generates the string of LaTeX code to compile for this font and digit'''
    preamble = """
    \\documentclass[12pt]{article}
    \\pagenumbering{gobble}
    \\usepackage{amsmath}
    \\usepackage{amssymb}
    """
    begin = """\\begin{document}"""
    
    if(font == "default"):
        content = """\\Huge{{${}$}}""".format(digit)
    elif(font == "euler"):
        preamble += "\\usepackage{concrete}"
        preamble += "\\usepackage{euler}"
        content = """${}$""".format(digit)
    elif(font == "typewriter"):
        content = """\\Huge{{$\\mathtt{{ {} }}$}}""".format(digit)
    elif(font == "roman"):
        content = """\\Huge{{$\\mathrm{{ {} }}$}}""".format(digit)
    elif(font == "serif"):
        content = """\\Huge{{$\\mathsf{{ {} }}$}}""".format(digit)
    elif(font == "bb"):
        preamble += """
        \\usepackage{bbold}
        """
        content = """\\Huge{{$\\mathbb{{ {} }}$}}""".format(digit)
    elif(font == "mathit"):
        preamble += """
        \\usepackage{amsmath}
        """
        content = """\\Huge{{$\\mathit{{ {} }}$}}""".format(digit)
    elif(font == "Antykwa"):
        preamble += """
        \\usepackage[math]{anttor}
        \\usepackage[T1]{fontenc}
        """
        content = """\\Huge{{$ {} $}}""".format(digit)
    elif(font == "mathpazo"):
        preamble += """
        \\usepackage{mathpazo}
        """
        content = """\\Huge{{$ {} $}}""".format(digit)
    elif(font == "boisik"):
        preamble += """
        \\usepackage{boisik}
        \\usepackage[OT1]{fontenc}
        """
        content = """\\Huge{{$ {} $}}""".format(digit)
    elif(font == "bakersville"):
        preamble += """
        \\usepackage[T1]{fontenc}
        \\usepackage{baskervillef}
        \\usepackage[varqu,varl,var0]{inconsolata}
        \\usepackage[scale=.95,type1]{cabin}
        \\usepackage[baskerville,vvarbb]{newtxmath}
        \\usepackage[cal=boondoxo]{mathalfa}
        """
        content = """\\Huge{{$ {} $}}""".format(digit)
    elif(font == "sanscomputer"):
        preamble += """
        \\usepackage{sansmathfonts}
        \\usepackage[T1]{fontenc}
        """
        content = """\\Huge{{$ {} $}}""".format(digit)
    elif(font == "neohellenic"):
        preamble += """
        \\usepackage[default]{gfsneohellenic}
        \\usepackage[LGR,T1]{fontenc} %% LGR encoding is needed for loading the package gfsneohellenic
        """
        content = """\\Huge{{$ {} $}}""".format(digit)
    elif(font == "Antywaka condensed"):
        preamble += """
        \\usepackage[condensed,math]{anttor}
        \\usepackage[T1]{fontenc}
        """
        content = """\\Huge{{$ {} $}}""".format(digit)
   
    elif(font == "custom"):
        preamble += """
        \\DeclareMathAlphabet{\\mathpzc}{OT1}{pzc}{m}{it}
        """
        content = """\\Huge{{$\\mathpzc{{ {} }}$}}""".format(digit)

    footer = """
    
    
    \\end{document}
    """

    return preamble + begin + content + footer



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
    
    #generate latex documents
    for i in range(n):
        font = random.choice(fonts)
        latex_doc = gen_latex(symbol, font)
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
        size0 = tmp.shape[0] + 2
        size1 = tmp.shape[1] + 2
        arr = np.ones((size0,size1))
        hw = maxarr - minarr + 1 #height, width
        remainder_height = (size0 - hw[0]) // 2 
        remainder_width = (size1 - hw[1]) // 2
        #tmp = image[minarr[0] : minarr, startcol : fincol]
        arr[remainder_height : hw[0] + remainder_height , remainder_width : hw[1] + remainder_width] = tmp  
        if(plot):
            plt.imshow(arr, cmap=plt.get_cmap('Greys_r'))
            plt.show()
        os.system("rm " + imgloc)
        plt.imsave(imgloc[:-4] +str(i)+".jpg" , arr, format="jpg", cmap=plt.get_cmap('Greys_r'))
    return arr



#r = list(range(33, 123)) #keyboard values of ascii table
r = [33, 43,45 ] + list(range(48,58)) + [60, 61, 62] + list(range(65,91)) + list(range(97,123))
r = [chr(x) for x in r] #remove special characters and escape characters
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





    
