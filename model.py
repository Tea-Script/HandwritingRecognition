
from datasets import *
from common_functions import *

class LevenshteinDist(nn.Module):
    def __init__(self):
        super(LevenshteinLoss, self).__init__()
    def forward(self, outputs, labels):
        return levenshtein_dist(outputs, labels)

def levenshtein_dist(pred, targets):
    '''preds are arrays of size classes with floats in them'''
    '''targets are arrays of all the classes from the batch'''
    '''we return the edit distance / length'''
    #pred = [class_names[x] for x in pred]
    return LEV.Levenshtein(pred, targets, simplify=False)


def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_lev_dist = float('inf')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batches = 0
            avg_lev_Dist = 0
            # Iterate over data.
            print("about to iterate over dataloader")
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = inputs.float()
                #print(inputs, labels)
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                #print(inputs)
                # forward
                
                outputs = model(inputs)
                #outputs = nn.functional.sigmoid(outputs)
                _, preds = torch.max(outputs, 1) 
                label = labels.diag().long()
                                
                #print(labels.shape)
                #print(pred.shape)
                
                loss = criterion(outputs, label)

                # backward + optimize only if in training phase
                if phase == 'train':
                    #print("backward step of training phase")
                    loss.backward()
                    optimizer.step()
                    #print("Optimizer adjusted")

                # statistics
                #print("calculating order statistics")
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds.data == label.data)
                avg_lev_Dist += levenshtein_dist(preds.data, label.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_lev_dist = avg_lev_Dist / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_lev_dist < best_lev_dist:
                #print("deepcopying model")
                best_acc = epoch_acc
                best_lev_dist = epoch_lev_dist
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Levenshtein Distance: {}'.format(best_lev_dist))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, len(class_names))

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=3)
torch.save(model_ft, "./weights1")

def visualize_model(model, num_images=9):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['test']):
        inputs, labels = data
        inputs = inputs.float()
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        __, preds = torch.max(outputs, 1) 
        #preds = nn.functional.sigmoid(preds).round()
        labels = labels.diag()
                
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//3, 3, images_so_far)
            ax.axis('off')
            #print(preds,j)
            ax.set_title('predicted: {}'.format(class_names[int(preds.data[j])]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)
print("visualizing model")
visualize_model(model_ft)

