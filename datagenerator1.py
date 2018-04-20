train_size = 30
test_size = 15
from skimage import io, filters
from common_functions import *


def preprocess(generate = False):
    if(generate): #if this is False, the samples have been generated
        clear_dir("./train/")
        clear_dir("./test/")
        train_samples = generate_samples(train_size, folder = "./train/", nm="")
        test_samples = generate_samples(test_size, folder = "./test/", nm="")
    else:
        train_samples = [imread(os.path.join("./train", x)) for x in os.listdir("./train/") if x[-4:] == ".jpg"]
        test_samples = [imread(os.path.join("./test", x)) for x in os.listdir("./test/") if x[-4:] == ".jpg"]
        
    #preprocessing
    for i in range(train_size):
        dir = './train/{}'.format(i)
        try:
            os.system("rm -r " + dir)
            os.system("mkdir  {}".format(dir))
        except Exception as e: print(e)
        
        img = train_samples[i]    
        boxes = get_bounding_boxes(train_samples[i])
        subdir = dir + '/{}'.format(len(boxes)) 
        j = 0
        for box in boxes:
            mini,maxi,minj,maxj = box
            image = img[mini:maxi+1, minj:maxj+1]
            #image = filters.gaussian(image, 5)
            #image = imresize(image, (56,56))
            if(image.shape[0] > 3 and image.shape[1] > 3):
                imsave(subdir + "{}.jpg".format(j), image)
            j += 1

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
            #image = filters.gaussian(image, 5)
            #image = imresize(image, (299, 299))
            if(image.shape[0] > 3 and image.shape[1] > 3):
                imsave(subdir + "{}.jpg".format(j), image)
            j += 1
if(__name__ == "__main__"):
    preprocess(generate=1)
