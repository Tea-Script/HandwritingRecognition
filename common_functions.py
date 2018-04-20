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

modify_level = ["^", "_", "\\frac", " ", "\\sqrt"]

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
            r = list(range(33, 123)) #keyboard values of ascii table
            blacklist = [91,92,93,94,95,35,36,37,38, 39]
            r = [x for x in r if x not in blacklist] #remove special characters and escape characters
            n = random.choice(r)
            c = chr(n)
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
    
def simplify(latex, verbose=0):
    '''Returns a list containing each character in a LaTex document string in order of appearance'''
    start = latex.find("\\begin{document}") + len("\begin{document}")
    end = latex.find("\\end{document}")
    latex = latex[start: end] #document body
    latex = latex.replace("\\[","") #removing noncharacter command for equations
    latex = latex.replace("\\]","") #removing noncharacter command for equations
    latex = latex.replace("\n", "")
    #latex = latex.replace("$", "") #removing noncharacter command for equations
    #latex = latex.replace("\text{") should make a function for this if necessary later on
    #find all supported LaTeX commands
    escaped_chars = [re.escape(x) for x in supported_characters]
    found_symbols = re.findall(r"(?=("+'|'.join(escaped_chars)+r"))", latex)
    found_symbols = list(filter(None, found_symbols)) #remove empty strings    
    #search latex file. For every "\" add the next found supported word to a list, then remove its first occurence from the string and list
    arr = []
    if(verbose): print(found_symbols)
    for c in latex:
        try:
            if(c == "\\"): #found a "\" command
                sym = found_symbols.pop(0)
                arr.append(sym)
                latex = latex.replace(sym, "", 1)
            elif(c in "{}^_" ): #found a position delimeter or container that doesn't belong to a command
                pass
            else: #found a normal character
                arr.append(c)
        except IndexError:
            if(verbose): print("Warning no symbols remaining")
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
        self.M = np.array([[-1]*(len(b)+1)]*(len(a)+1))
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
LEV = lev()
def test(function, verbose=True):
    '''Used to test the various function implementations before proceeding with the project'''
    if(function == "gen_random_latex"):
        try: #if this throws an error, the latex did not successfully compile, otherwise, it was successful
            gen_random_latex()
            if(verbose): 
                print("Test Successful")
        except:
            if(verbose):
                print("Test Failed")
                print(sys.exc_info()[0])
    if(function == "simplify"):
        gen_random_latex()
        with open("rand0.tex") as f:
            l = f.read().replace("\n","")
        if(verbose):
            print("Compare the following result to the latex file to test")
            print(simplify(l))
    if(function == "Levenshtein"):
        arr1 = ["1", "3", "2", "5", "10"]
        arr2 = ["1", "2", "9" ]
        bul2 = LEV.Levenshtein(arr1,arr2) == 3
        arr1 = ["11", "3", "2", "5", "10"]
        arr2 = ["1", "2", "9", "5", "5", "6" ]
        bul1 = LEV.Levenshtein(arr1,arr2) == 5
        if(verbose):
            if(bul1 and bul2):
                print("Both Levenshtein tests were successful")
            else:
                print("One or more tests were unsuccessful")
    if(function == "generate_samples"):
        if(verbose):
            print(generate_samples(20)[0])
def clear_dir(dir = "./"):
    '''Removes jpg, log, aux, and tex files from directory'''
    try:
        os.system("rm  {0}*.tex {0}*.aux {0}*.log {0}*.jpg {0}*.pdf".format(dir))
        #if(dir != "./"):
    except Exception as e: print(e)


#0 == black
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
    
def find_connected3(img, i, j, component_num, components):
    if(img[i,j] < 1 and components[i,j] == 0):
        components[i,j] = component_num 
    if(img[i,j] < 1):
        if(img[i, j+1] < 1): components[i,j] = component_num
def get_bounding_boxes2(img, plot= False):
    #this algorithm is based on the one from appendix B of the dissertation
    labels = np.zeros(img.shape, dtype = np.int16)
    eq_table = [0]
    currentlabel = 0
    used_labels = [False]

    #first pass through
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if(img[i,j] < 1):
                neigh = [0]*4
                nbnei = 0;
                if(labels[i - 1][j] > 0):    
                    neigh[nbnei] = labels[i - 1][j]
                    nbnei += 1
                if(labels[i+1][j] > 0):
                    neigh[nbnei] = labels[i + 1][j]
                    nbnei += 1
                if(labels[i][j+1] > 0):
                    neigh[nbnei] = labels[i][j + 1]
                    nbnei += 1
                if(labels[i][j-1] > 0):
                    neigh[nbnei] = labels[i][j - 1]
                    nbnei += 1
                if(nbnei == 0): #create new label
                    currentlabel += 1
                    eq_table.append(currentlabel)
                    used_labels.append(False)
                    labels[i][j] = currentlabel
                else:
                    minlabel = img.shape[0] * img.shape[1]
                    for i in range(nbnei):
                        if(neigh[i] < minlabel):
                            minlabel = eq_table[neigh[i]]
                    labels[i][j] = eq_table[minlabel]
                    for i in range(nbnei):
                        if(eq_table[neigh[i]] > minlabel):
                            eq_table[neigh[i]] = eq_table[minlabel]
                            
    for i in range(len(eq_table)):
        if(eq_table[i] > eq_table[eq_table[i]]):
            eq_table[i] = eq_table[eq_table[i]]
            
    #second pass through
    bounding_boxes = np.zeros((currentlabel + 1, 4), dtype = np.int16)
    newlab = -1
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if(labels[i,j] > 0):
                newlab = eq_table[labels[i,j]]
                if(used_labels[newlab]): #update bounding box
                    if(bounding_boxes[newlab][ 0 ] > i): bounding_boxes[newlab][0] = i
                    elif(bounding_boxes[newlab][1] < i): bounding_boxes[newlab][1] = i
                    if(bounding_boxes[newlab][ 2 ] > j): bounding_boxes[newlab][2] = j
                    elif(bounding_boxes[newlab][3] < j): bounding_boxes[newlab][3] = j
                
                else:
                    used_labels[newlab] = True
                    bounding_boxes[newlab][0],bounding_boxes[newlab][1] = i,i
                    bounding_boxes[newlab][2],bounding_boxes[newlab][3] = j,j

    if(plot):
        for quad in bounding_boxes:
            mini,maxi,minj,maxj = quad
            img[mini:maxi, minj:maxj] = 0
        imsave("boxed.jpg",img)
        plt.imshow(img, cmap=plt.get_cmap('Greys_r'), aspect = "auto")
        plt.show()
    return bounding_boxes
    
                
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

if __name__ == "__main__":
	test("gen_random_latex")
	test("Levenshtein")
	test("simplify", verbose= 1)
	test("generate_samples", verbose = 1)
	boxes = get_bounding_boxes(generate_samples(1)[0], plot=1)



