from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classification_report(outputs, labels):

    '''
    report is in batches inorder to save memory usage.
    The function gives various matrics performance of the model
    :param predictions: predictions by the model
    :param labels: original labels
    '''
    predictions = torch.max(outputs, 1)[1].to(device)
    
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = accuracy_score(predictions, labels)
    recall = recall_score(predictions, labels)
    f1 = f1_score(predictions, labels)
    cm = confusion_matrix(predictions, labels)

    print("Accuracy : {}, Recall : {}, F1 Score : {}".format(accuracy, recall, f1))
    print("************************* Confusion Matrix ***********************")
    print(cm)


def evaluate_model(model, test_loader):
    '''

    :param cnn_model: trained model
    :param test_loader: test dataset loader
    '''
    count = 1
    with torch.no_grad():
        correct = 0
        total_samples = 0
        for images, labels in test_loader:
            
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            
            print("Report on BATCH: ", count)
            classification_report(output, labels)
            count +=1
            
            
def accuracy(outputs, labels):
    '''

    :param outputs: predictions from the model
    :param labels: original labels
    :return: correct: total number of correct classifications
    '''
    predictions = torch.max(outputs, 1)[1].to(device)
    correct = (predictions == labels).sum().to(device)
    
    acc = correct/len(predictions)
    acc = acc.cpu().numpy()
    return acc
  
