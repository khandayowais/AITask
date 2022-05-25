import torch
from torchvision import datasets, transforms
from os import getcwd
import numpy as np
from model import CNNModel
import torch.optim as optim
from eval_utils import evaluate_model, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer(EPOCHS=5, BATCH_SIZE=128):
    """

    :param PATH: PATH of the dataset
    :param EPOCHS: number of epochs default 1
    :param BATCH_SIZE: batch size default 256
    """
    # PATH

    train = datasets.ImageFolder(getcwd() + '/SyntheticImages/train/',
               transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
    
    
    n = len(train) 
    n_test = int(0.1 * n) 
    indexes = torch.randperm(n)
    validation = torch.utils.data.Subset(train, indexes[:n_test])  
    train = torch.utils.data.Subset(train, indexes[n_test: ])
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)  
    val_loader = torch.utils.data.DataLoader(validation , batch_size=BATCH_SIZE, shuffle=True)
    
    test = datasets.ImageFolder(getcwd() + '/task2/test/', transforms.Compose([transforms.Grayscale(),
                                                                               transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test, batch_size=20)
    
    print(len(test),getcwd())
    model = CNNModel()  # init model
    model.to(device)

    error = torch.nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    epoch = 0

    train_losses = []
    train_acces = []

    val_acces = []
    val_losses = []

    # early_stopping =
    print('*************** Model Training Started ************** ')
    model.train()
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
        
        # , train_acc, val_loss, val_acc)
        train_loss = round(np.average(train_losses), 4)
        train_acc = round(np.average(train_acces), 4)
        val_loss = round(np.average(val_losses), 4)
        val_acc = round(np.average(val_acces), 4)

        epoch += 1

        # Printing the model Traning Accuracy and Testing Accuracy Trining Loass and Validation Loss
        
        print("Epoch: {}, Train Loss: {}, Val Loss: {}, Train Accuracy: {}, Val Accuracy: {}".format(epoch, train_loss,
                                                                                                     val_loss,
                                                                                                     train_acc,
                                                                                              val_acc))
        
        #print("Epoch: {}, Train Loss: {}, Val Loss: {}".format(epoch, train_loss, train_acc))




    print('*************** Model Training Finished ************** ')
    print('*************** Testing Model on the Test Data ************** ')

    evaluate_model(model, test_loader)

trainer(100)
