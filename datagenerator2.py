train_size = 30
test_size = 15
from skimage import io, filters
from common_functions import *


def preprocess(generate = False):
    if(generate): #if this is False, the samples have been generated
        clear_dir("./handtrain/")
        clear_dir("./handtest/")
        train_samples = generate_special_samples(train_size, folder = "./handtrain/", nm="")
        test_samples = generate_special_samples(test_size, folder = "./handtest/", nm="")
    else:
        train_samples = [imread(os.path.join("./handtrain", x)) for x in os.listdir("./handtrain/") if x[-4:] == ".jpg"]
        test_samples = [imread(os.path.join("./handtest", x)) for x in os.listdir("./handtest/") if x[-4:] == ".jpg"]
        
    #preprocessing
    for i in range(train_size):
        dir = './handtrain/{}'.format(i)
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
        dir = './handtest/{}'.format(i)    
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
