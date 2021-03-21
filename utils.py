import numpy as np
import time
from collections import OrderedDict
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models, transforms
from dataloader import Lung_Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import os
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = cm.astype(int)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def validation(model, testloader, criterion, device):
    model.eval()
    model.to(device)
    test_loss = 0
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        
        # revert one hot encoding
        labels = torch.argmax(labels, dim=1)
        output = model.forward(images)
        
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

def evaluation(model, testloader, device='cuda'):
    model.eval()
    model.to(device)

    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        
        # revert one hot encoding
        labels = torch.argmax(labels, dim=1)
        output = model.forward(images)
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return accuracy/len(testloader)

def confusionMatrix(model, testloader, nb_classes=3, device='cuda'):
    model.to(device)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(testloader):
            inputs = inputs.to(device)
            classes = torch.argmax(classes, dim=1)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

#     print(confusion_matrix)
    return confusion_matrix

def saveHistorical(dir, save_dir):
    # save history
    file_name = f"{save_dir}/history.pkl"
    with open(file_name,"wb") as f:
        pickle.dump(history, f)
        
def save(model, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(),os.path.join(save_dir, name))

def load(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def plot_loss(training_loss, validation_loss, path):
    epochs = len(training_loss)
    plt.plot(range(1,epochs+1), training_loss, 'g', label='Training loss')
    plt.plot(range(1,epochs+1), validation_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.show()
    
def train(model, n_epoch, criterion, optimizer, trainloader, validloader, model_name, save_dir, save_model=True, device='cuda'):
    model.to(device)
    start = time.time()
    
    epochs = n_epoch
    steps = 0
    running_loss = 0
    running_accuracy = 0
    
    print_every = 100
    
    history = {}
    history["training_loss"]=[]
    history["validation_loss"]=[]
    
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            
            # revert one hot encoding
            labels = torch.argmax(labels, dim=1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            running_accuracy += equality.type(torch.FloatTensor).mean()
            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)
                    
                training_loss=running_loss/print_every
                validation_loss=test_loss/len(validloader)
                
                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Training Loss: {:.3f} - ".format(running_loss/print_every),
                      "Training Accuracy: {:.3f} - ".format(running_accuracy/print_every),
                      "Validation Loss: {:.3f} - ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                running_accuracy = 0
                # Make sure training is back on
                model.train()
        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        # save
        if save_model:
            save(model, save_dir, model_name)
        
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    return {"model":model, "history":history}