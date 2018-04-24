import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imsave

class f:
    def __init__(self):
        self.M =  np.zeros(img.shape)

    def find_connected(self, img, i,j): #Memoized
        '''recursively generates list of all neighbors of i and j, black is a set'''
        BOUND = 56 #assume a character is within a BOUND x BOUND array
        if((max(0, i - BOUND) < i < min(img.shape[0], i + BOUND)) and (max(0, j - BOUND) < j < min(img.shape[1], j + BOUND))): 
            if(self.M[i,j]): return 1
            elif(img[i,j] < 1):
                self.M[i,j] = 1
                #check cardinal directions, as all others will be recursively detected   
                self.find_connected(img, i, j+1)
                self.find_connected(img, i+1, j)
                self.find_connected(img, i, j-1)
                self.find_connected(img, i-1, j)
                return 1
            return 0
    def get_indices(self):
        return np.argwhere(self.M)

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
            #M = f()
            pairs = np.array(labels).flatten()
            if(img[i,j] < 1 and (i,j) not in pairs): #if component black find its neighbors
                #M.find_connected(img, i,j)
                #conn = M.get_indices()
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


img = imread("sample.jpg", mode="L")/255

get_bounding_boxes(img, plot=True)

