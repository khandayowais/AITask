import numpy as np
import torch
import torch.optim as optim
from model import CNNModel
from dataset_loader import dataset_load
from os import getcwd
from eval_utils import evaluate_model, accuracy
import os
#from pytorchtools import EarlyStopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer(PATH, EPOCHS=5, BATCH_SIZE=500):
    """

    :param PATH: PATH of the dataset
    :param EPOCHS: number of epochs default 1
    :param BATCH_SIZE: batch size default 256
    """
    # PATH

    train_loader = dataset_load(PATH=PATH, KIND='train', BATCH_SIZE=BATCH_SIZE)
    val_loader = dataset_load(PATH=PATH, KIND='validation', BATCH_SIZE=BATCH_SIZE)
    test_loader = dataset_load(PATH=PATH, KIND='test', BATCH_SIZE=BATCH_SIZE*8)


    model = CNNModel()  # init model
    model.to(device)

    error = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch = 0


    train_losses = []
    train_acces = []
    
    val_acces = []
    val_losses = []
    


    #early_stopping =
    print('*************** Model Training Started ************** ')
    
    for epoch in range(EPOCHS):
        total_correct = 0
        train_loss = 0
        
        model.train()
        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            train_losses.append(loss.item())
            train_acces.append(accuracy(outputs, labels))
        # Validation of the model

        model.eval()
        for images, labels in val_loader:
                               
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = error(outputs, labels)
            val_losses.append(loss.item())
            val_acces.append(accuracy(outputs, labels))

        
        
        #, train_acc, val_loss, val_acc)
        train_loss = round(np.average(train_losses),4)
        train_acc = round(np.average(train_acces), 4)
        val_loss = round(np.average(val_losses), 4)
        val_acc = round(np.average(val_acces), 4)
        


        epoch += 1

        # Printing the model Traning Accuracy and Testing Accuracy

        print("Epoch: {}, Train Loss: {}, Val Loss: {}, Train Accuracy: {}, Val Accuracy: {}".format(epoch, train_loss, val_loss, train_acc, val_acc))
    

    print('*************** Model Training Finished ************** ')
    print('*************** Testing Model on the Test Data ************** ')
   
    evaluate_model(model, test_loader)
    
    print('*************** Saving the Trained Model ************** ')
    
    path_to_saved_model = getcwd() +  '/PretrainedModel/'
    
    # checking if directory exists if not create one
    if not os.path.exists(path_to_saved_model):
        os.mkdir(path_to_saved_model)
    
    #assigning the model name accoring to train times
    version = 1
    while(True):
        
        name = 'modelv'+str(version) + '.pth'
        model_name = path_to_saved_model +  name
        if os.path.exists(model_name):
            version += 1
        else:
            break
        
   
    torch.save(model.state_dict(),  model_name)
    
    print('*************** Model Saved Sucessfully ************** ')
   





