#TODO: Segementation
#TODO: Sample Generation with Hand (last step, since we can then compare this to not using Hand)
#Complete: Sample Generation without Hand
#Complete: Create function that takes Latex and turn it into a list of ordered characters with no formatting
#TODO: Load Pretrained Neural Network
#Complete: Levenshtein Distance

#Assume Segmentation has already been completed.
#Now I have a series of different sized character boxes for each shape. 
#Standardize shape by making all of them 56x56 numpy arrays of intensity
import re
import random
import subprocess
from scipy.misc import *
import sys
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io

warnings.filterwarnings("ignore", category=DeprecationWarning) #ignore imread deprecation warning


random.seed(1111)

with open('latexsymbols.txt', 'r') as f:
    supported_characters = f.read().split('\n')

modify_level = ["^", "_", " "]#, "\\frac", "\\sqrt"]

def gen_random_latex(hand=False, name="rand0", folder="./"):
    name = folder + name
    latexheader = """
        \\documentclass[12pt]{article}
        \\pagenumbering{gobble}
        \\usepackage{amsmath}
        \\usepackage{amssymb}
        \\addtolength{\\topmargin}{-1.5in}
        \\begin{document}\n
        \\begin{minipage}[t][0pt]{\\linewidth}\n
        """
    #generates one latex document with random values and levels
    num_open = 0 
    open_queue = 0 #add open bracket after next closing bracket if positive
    body = "\\["
    linecontent = False #are all \[ \] full?
    bracketcontent = False #are all {} full?
    supportchar = "" #is there a space after the last support char, if the next char is not ^ or _
    space = True #is the last character white space
    charlast = True #was the last character ascii/support?
    for i in range(350):
        x = random.random()
        if(num_open == 0 and open_queue == 0):
            bracketcontent = False
        if(x < .4 and num_open > 0 and bracketcontent): #If open bracket, close it 40ish% of the time
            body += "}"
            if(open_queue > 0):
                body += "{"
                open_queue -= 1
            else:
                num_open -= 1
            charlast = False
            bracketcontent = False
        if(x < .01 and charlast and not space): # apparently ' is considered a superscript
            c = "'"
            charlast = False
            bracketcontent = False
        if(x < .09 and charlast and not space): #choose a ^ or _ or frac or space 5% of the time
            c = random.choice(modify_level)
            num_open += 1
            body += c + "{"
            if(c == "\\frac"):
                open_queue += 1
            if(c == " "):
                space = True
            else:
                space = False
            supportchar = ""
            bracketcontent = False
            charlast = False
        elif(x < .3): #choose a supported character 35% of time
            c = random.choice(supported_characters)
            body += c
            linecontent = True
            if(num_open):
                bracketcontent = True
            supportchar = " "
            space = False
            charlast = True
        elif(x < .38 and num_open == 0 and open_queue == 0 and linecontent): # new line 10ish% of time
            body += "\\]\n\\["
            linecontent = False
            supportchar = ""
            space = False
            charlast = False
            bracketcontent = False
        else: #choose random ASCII on standard keyboard 50% of time
            blacklist = [91,92,93,94,95,35,36,37,38, 39]
            r = [33, 43,45 ] + list(range(48,57)) + [60, 61, 62] + list(range(65,91)) + list(range(97,123))
            r = [chr(x) for x in r if x not in blacklist] #remove special characters and escape characters
            c = random.choice(r)
            body += supportchar + c #add a space before c if previous char is escaped
            linecontent = True
            space = False
            charlast = True
            if(num_open):
                bracketcontent = True
    while(num_open > 0): #If open bracket, close it 40ish% of the time
            body += " e}"
            if(open_queue > 0):
                body += "{r"
                open_queue -= 1
            else:
                num_open -= 1 
    latexend = """
        \\]\n\\end{minipage}\n\\end{document}
        """
    
    #generate latex documents
    latex_doc = latexheader + body + latexend
    f = open("{}.tex".format(name),"w+")
    f.write(latex_doc)
    f.close()
    return latex_doc

def clear_dir(dir = "./"):
    '''Removes jpg, log, aux, and tex files from directory'''
    try:
        os.system("rm  {0}*.tex {0}*.aux {0}*.log {0}*.jpg {0}*.pdf".format(dir))
        #if(dir != "./"):
    except Exception as e: print(e)



def get_latex_img(name, folder="./"):
    name = folder + name
    #compile latex documents
    #dependency: texlive; bash
    subprocess.check_output(['pdflatex', "-output-directory=" + folder,  '{}.tex'.format(name)])
    #convert pdfs to images jpg
    #dependency: ImageMagick; bash
    subprocess.check_output(['convert', '-quality', '100',  '{}.pdf'.format(name), '{}.jpg'.format(name)])
    #now we need to read the images in as arrays and segment them into 28x28px arrays consisting of all the black squares.
    #dependency: Scipy
    img = imread("{}.jpg".format(name), mode="L")/255 #read in latex doc as image
    return img
def generate_samples(num, hand=False, folder="./", nm="rand"):
    '''generates num latex docs and returns the set of images'''
    imgs = []
    for i in range(num):
        name = nm + str(i)
        gen_random_latex(hand, name=name, folder=folder)
        img = np.round(get_latex_img(name, folder=folder))
        imgs.append(img)
    return imgs
   
def find_connected(img, i,j, black): #issue: takes extraneous amount of time and can't see whole characters
    '''recursively generates list of all neighbors of i and j, black is a set'''
    #print(i,j)
    if((0 < i < img.shape[0] and 0 < j < img.shape[1])): 
        
        if(img[i,j] < 1 and (i,j) not in black):
            #print(i,j)
            black.add((i,j))
            #check cardinal directions, as all others will be recursively detected   
            return find_connected(img, i, j+1, black) | find_connected(img, i+1, j, black) \
                | find_connected(img, i, j-1, black) | find_connected(img, i-1, j, black)
    return black
    
def find_connected2(img, i, j, black): #issue takes a lot of time and cant see whole characters
    BOUND = 28 #assume a character is within a BOUND x BOUND array
    if((max(0, i - BOUND) < i < min(img.shape[0], i + BOUND)) and (max(0, j - BOUND) < j < min(img.shape[1], j + BOUND))): 
        if(img[i,j] < 1 and (i,j) not in black):
            #print(i,j)
            black.add((i,j))
            #check cardinal directions, as all others will be recursively detected   
            return find_connected(img, i, j+1, black) | find_connected(img, i+1, j, black) \
                | find_connected(img, i, j-1, black) | find_connected(img, i-1, j, black)
    return black
    
                
def get_bounding_boxes(img, plot = False):
    '''labels each connected component in the img, then finds the bounding box for that component'''
    '''We want to find all indices that communicate with i and j and add them to the same component'''
    if(plot):
        imsave("og.jpg",img)
    labels = []
    lowest = 0
    #print(img.shape)
    #print(img)
    #print("searching for components")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #detecting connected components
            #print(lowest)
            pairs = np.array(labels).flatten()
            if(img[i,j] < 1 and (i,j) not in pairs): #if component black find its neighbors
                #print("found a component")
                black = set()
                conn = find_connected2(img, i,j, black)
                labels.append(conn)
    
    
    
    bounding_boxes = set()
    #print(labels)
    for component in labels:
        #print(component)
        mini = img.shape[0] #find all four values in one pass through the component
        minj = img.shape[1]
        maxi = 0
        maxj = 0
        for pair in component:
            if(pair[0] < mini): mini = pair[0]
            if(pair[0] > maxi): maxi = pair[0]
            if(pair[1] < minj): minj = pair[1]
            if(pair[1] > maxj): maxj = pair[1]
        bounding_boxes.add((mini, maxi, minj, maxj))
        img2 = img.copy()
        img2[mini:maxi, minj:maxj] = 0 #drawing black box over character
    #print("Preparing to plot bounding boxes")
    if(plot):
        imsave("boxed.jpg",img2)
        plt.imshow(img2, cmap=plt.get_cmap('Greys_r'), aspect = "auto")
        plt.show()
    return list(bounding_boxes)
#boxes = get_bounding_boxes(generate_samples(1)[0], plot=0)
#chars = []
#for box in boxes:
#    mini,maxi,minj,maxj = box
#    chars.append(imresize(img[mini:maxi, minj:maxj]), (56,56)) #resizes all images to 56x56 for ML

test_size = 15
from skimage import io, filters


def preprocess(generate = False):
    if(generate): #if this is False, the samples have been generated
        clear_dir("./test/")
        test_samples = generate_samples(test_size, folder = "./test/", nm="")
    else:
        test_samples = [imread(os.path.join("./test", x)) for x in sorted(os.listdir("./test/")) if x[-4:] == ".jpg"]
    
    for i in range(test_size):
        dir = './test/{}'.format(i)    
        try:
            os.system("rm -r" + dir)
            os.system("mkdir  {}".format(dir))
        except Exception as e: print(e)
        img = test_samples[i]
        boxes = get_bounding_boxes(test_samples[i])
        subdir = dir + '/{}'.format(len(boxes)) 
        j = 0
        for box in boxes:
            mini,maxi,minj,maxj = box
            image = img[mini:maxi+1, minj:maxj+1]
            if(image.shape[0] > 3 and image.shape[1] > 3):
                imsave(subdir + "{}.jpg".format(j), image)
            j += 1
    
if(1):
    preprocess(generate=1)
