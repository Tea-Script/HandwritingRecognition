
from datasets import *
from confusionmeter import *

print(dataloaders["test"])
print(class_names)
def run_model(model):
    model.train = False
    best_corrects = 0
    top3_best_corrects = 0
    individual_accuracy = [0]*len(class_names)
    confusion_matrix = ConfusionMeter(len(class_names))
    for data in dataloaders["test"]:
        inputs, labels = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        outputs = model(inputs)
        #print(outputs)
        _, preds = torch.max(outputs.data, 1) 
        _, allpreds = torch.sort(outputs.data, descending=True, dim=1)
        #print(allpreds.numpy())
        top3preds = allpreds[:3]
        best_corrects += torch.sum(preds == labels.data)
        #top3_best_corrects += torch.sum(labels.data in top3preds) 
        confusion_matrix.add(preds, labels.data)

    confusion_matrix = confusion_matrix.value()
    TP = np.diag(confusion_matrix)
    FP = np.sum(np.sum(confusion_matrix, axis=0) - TP)
    FN = np.sum(np.sum(confusion_matrix, axis=1) - TP)
    TP = np.sum(TP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    best_accuracy = best_corrects / dataset_sizes["test"]
    #top3_accuracy = top3_best_corrects / dataset_Sizes["test"]
    
    print("num correct", best_corrects, "num true postives",TP,"num false positives", FP,"num false negatives", FN)
    print("precision", precision)
    print("recall", recall)
    print("accuracy", best_accuracy)
    #print(top3_accuracy)


def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
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
            # Iterate over data.
            print("about to iterate over dataloader")
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
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
                # forward
                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1) 
                #preds = preds.float()
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                #print("calculating order statistics")
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                #print("deepcopying model")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#model_ft = models.resnet50(pretrained=True)
model_ft = models.densenet121(pretrained="imagenet")
for param in model_ft.parameters():
    param.requires_grad = False

#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, len(class_names))
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, len(class_names))


if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.SGD(model_ft.classifier.parameters(), lr=0.2, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=8)

torch.save(model_ft.state_dict(), "./weights.pt")
torch.save(model_ft.state_dict(), "../Model1/weights.pt")
model_ft.load_state_dict(torch.load("./weights.pt"))
run_model(model_ft)


def visualize_model(model, num_images=9):
    #was_training = model.training
    #model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['test']):
        inputs, labels = data
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        __, preds = torch.max(outputs, 1) 

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//3, 3, images_so_far)
            ax.axis('off')
            #print(preds,j)
            ax.set_title('predicted: {}'.format(class_names[int(preds.data[j])]))
            imshow(inputs.cpu().data[j])
            inp = inputs.cpu().data[j]
            inp = inp.numpy().transpose((1, 2, 0))
            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)
print("visualizing model")
visualize_model(model_ft)

