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
import cv2
from skimage.transform import radon

warnings.filterwarnings("ignore", category=DeprecationWarning) #ignore imread deprecation warning


random.seed(2222)

with open('latexsymbols.txt', 'r') as f:
    supported_characters = f.read().split('\n')

modify_level = ["^", "_", " "]#, "\\frac", "\\sqrt"]


def gen_hand_latex(name="rand0", folder="./"):
    return

def gen_random_latex(hand=False, name="rand0", folder="./"):
    if(hand): return gen_hand_latex(name, folder)
    name = folder + name
    latexheader = """
        \\documentclass[12pt]{article}
        \\pagenumbering{gobble}
        \\usepackage{amsmath}
        \\usepackage{amssymb}
        \\usepackage{euler}
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
    for i in range(250):
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
        if(x < .09 and charlast and not space and list(body)[-1] != "}"): #choose a ^ or _ or frac or space 5% of the time
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
            r = [33, 43,45 ] + list(range(48,58)) + [60, 61, 62] + list(range(65,91)) + list(range(97,123))
            r = [chr(x) for x in r] #remove special characters and escape characters
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

def clear_dir(dir = "./test/"):
    '''Removes jpg, log, aux, and tex files from directory'''
    try:
        os.system("rm -r {}*".format(dir))
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
   


def get_bounding_boxes(file_name, out_dir):
    img = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1,1), np.uint8)
    imav = cv2.dilate(imgray,kernel, iterations=3)
    
    ret, thresh = cv2.threshold(imav, 127, 255, 0)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img, contours, -1, (0,255,0), 3)
    images = []
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) #sort images by x value
    for contour in sorted_contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)
        index = str(y) + str(x)        
        if(w > 2 and h > 2 and (x != 0 and y != 0)):
            # draw rectangle around contour on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.imshow("s",img)
            cv2.waitKey(100)
            images.append([x,y,w,h])
        

    # write original image with added contours to disk
    cv2.imwrite('boundingboxes.jpg', img)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()

    img = cv2.imread('boundingboxes.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((19,19), np.uint8)
    imav = cv2.erode(imgray,kernel, iterations=3)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow("s",thresh) 
    #cv2.waitKey(10000)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    lines = []
    #sort contours
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1]) #sort lines by y value 

    for contour in sorted_contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)
        index = str(y) + str(x)        
        if(w > 14 and h > 5 and (x!=0 and y!= 0)):
            # draw rectangle around contour on original image
            cv2.rectangle(img_final, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.imshow("s",img_final)
            cv2.waitKey(1000)
            linebox = [x,y,w,h]
            lines.append(linebox)
    # write original image with added contours to disk
    cv2.imwrite('boundinglines.jpg', img_final)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()
    final_order = [[]]*len(lines)
    index = "0"
    img = cv2.imread(file_name)
    for i,linebox in enumerate(lines):
        xl,yl,wl,hl = linebox
        for image in images:
            x,y,w,h = image
            if((yl <= y <= yl+hl or yl <= y+h <= yl + hl )):
                #image belongs in this linebox
                cropped = img[y :y +  h , x : x + w]
                final_order[i].append(cropped)
                s = index
                cv2.imwrite(os.path.join(out_dir, s) + ".jpg", cropped)
                index += "0"
                images.remove(image)
     
        
        
        
    return final_order


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
        get_bounding_boxes(dir+".jpg", dir)
        
        




if(__name__ == "__main__"):
    preprocess(generate=1)
    #get_bounding_boxes("test/0.jpg", "./test")
