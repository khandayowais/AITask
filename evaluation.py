from model import CNNModel
import torch
from os import getcwd
from eval_utils import classification_report
from dataset_loader import dataset_load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(MODEL_PATH,DATA_DIR, BATCH_SIZE):
    '''

    :param PATH: Path of the test datase
    :param BATCH_SIZE: batch size
    '''

    
    test_data_loader = dataset_load(PATH=DATA_DIR, KIND='t10k', BATCH_SIZE=BATCH_SIZE)
    
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print('****************** Pretrained Model Loaded Sucessfully ******************')
    
    count = 1
    with torch.no_grad():
        for images, labels in test_data_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images).to(device)
            
            print('*************** Report on BATCH {} **************** '.format(count))

            classification_report(output, labels)
            count = count + 1


