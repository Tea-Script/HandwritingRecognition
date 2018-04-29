from datasets import *
from scipy.misc import imread

r = [33,43,45 ] + list(range(48,58)) + [60, 61, 62] + list(range(65,91)) + list(range(97,123))
r = [chr(x) for x in r] #remove special characters and escape characters

class_names1 = r + supported_characters
class_names1 = [x for x in class_names1 if x not in ["", " ", '\n']]
class_names2 = [replace(x) for x in class_names1 if replace(x) not in ["", " "]]
print(len(class_names1), len(class_names2))
unescape = dict(zip(class_names2, class_names1))
class_names2 = list(sorted(class_names2))
print(class_names2)


model_ft = torchvision.models.densenet161(pretrained='imagenet')
num_ftrs = model_ft.classifier.in_features
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.classifier = nn.Linear(num_ftrs, len(class_names2))


model_ft.load_state_dict(torch.load("./weights.pt"))
model_ft.eval()

def get_symbol(image, model):
    
    img = data_transforms(image)
    inputs = Variable(img.unsqueeze(0)) 
    outputs = model(inputs)
    _, pred = torch.max(outputs.data, 1)
    #print(outputs)
    pred = int(pred[0]) 
    symbol = class_names2[pred]
    #print(symbol)
    symbol = unescape[symbol]
    return symbol    

def run_model(model):	
    
    n = 0
    avg_lev = 0
    acc = 0
    l_acc = 0
    for folder in sorted(os.listdir("test")):
        path = os.path.join("test", folder)
        if(os.path.isdir(path)):
            latex = []
            label = simplify(path + ".tex") 
            for f in sorted(os.listdir(path)):
                img = imread(os.path.join(path, f), mode="RGB")
                symbol = get_symbol(img, model)
                latex.append(symbol)
    
            '''for i, data in enumerate(dataloaders['test']):
        inputs, labels = data
        inputs = Variable(inputs)

        outputs = model(inputs)
        #label = simplify()
        __, preds = torch.max(outputs, 1) 
        #print(label)
        symbol = class_names2[int(preds[0])]
        print(symbol)
            '''
            print(latex[:20])
            print(label[:20])
            print(len(label), len(latex))
            
            l = LEV.Levenshtein(latex, label)
            avg_lev += l
            l_acc += l/len(label)
            latex = latex[:len(label)]
            label = label[:len(latex)]
            acc += np.sum(np.array(latex) == np.array(label))/ len(label)
            n += 1
            print("Levenshtein Distance of Sample is ", l)
            print("Accuracy by sample {0} is {1}".format(n, acc/n))
            print("Levenshtein Accuracy by Sample {} is {}".format( n, l_acc/n))
            print("Average Levenshtein Distance by sample {0} is {1}".format(n, avg_lev/n))
                
    avg_lev /= n 
    acc /= n
    l_acc /= n
    print("Average Levenshtein Distance of Model", avg_lev)
    print("Accuracy of model:", acc)
    print("Levenshtein accuracy of model:", l_acc)
run_model(model_ft)
        

