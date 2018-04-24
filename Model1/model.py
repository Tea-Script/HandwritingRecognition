from datasets import *
from scipy.misc import imread

r = [33, 43,45 ] + list(range(48,57)) + [60, 61, 62] + list(range(65,91)) + list(range(97,123))
r = [chr(x) for x in r] #remove special characters and escape characters

class_names1 = r + supported_characters
class_names2 = [replace(x) for x in class_names1 if replace(x) not in ["", " "]]
unescape = dict(zip(class_names2, class_names1))
class_names2 = list(sorted(class_names2))


model_ft = torchvision.models.densenet161(pretrained='imagenet')
num_ftrs = model_ft.classifier.in_features
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.classifier = nn.Linear(num_ftrs, len(class_names1))

model_ft = torch.load("weights2")

def get_symbol(image, model):
    
    img = data_transforms(image)
    inputs = Variable(img.unsqueeze(0)) 
    outputs = model(inputs)
    _, pred = torch.max(outputs.data, 1)
    #print(pred)
    pred = int(pred[0]) 
    symbol = class_names2[pred]
    #print(symbol)
    symbol = unescape[symbol]
    return symbol    

def run_model(model):	
    model.train = False
    n = 0
    avg_lev = 0
    acc = 0
    for folder in sorted(os.listdir("test")):
        path = os.path.join("test", folder)
        if(os.path.isdir(path)):
            latex = []
            label = simplify(path + ".tex") 
            for f in sorted(os.listdir(path)):
                img = imread(os.path.join(path, f), mode="RGB")
                symbol = get_symbol(img, model)
                latex.append(symbol)
            avg_lev += LEV.Levenshtein(latex, label)
            latex = latex[:len(label)]
            label = label[:len(latex)]
            print(len(latex),len(label))
            acc += np.sum(np.array(latex) == np.array(label))/ len(label)
            n += 1
            print("Accuracy by sample {0} is {1}".format(n, acc/n))
            print("Average Levenshtein Distance by sample {0} is {1}".format(n, avg_lev/n))
                
    avg_lev /= n 
    acc /= n
    print("Average Levenshtein Distance of Model", avg_lev)
    print("Accuracy of model:", acc)
run_model(model_ft)
        

