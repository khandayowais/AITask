import argparse
from os import getcwd
import os
from train import trainer
from evaluation import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=os.getcwd()+'/Data/')
    parser.add_argument('--function', type=str, default='train')
    parser.add_argument('--pretrained_model_path',type=str, default=getcwd()+'/PretrainedModel/')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--model', type=str, default = 'modelv1.pth')
    parser.add_argument('--epochs', type=int, default = 5)

    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = getcwd() + '/Data/'
    if not os.path.isdir(args.data_path):
        print("No data Directory")
    else:
        if args.function == 'train':
            trainer(PATH=args.data_path, EPOCHS=args.epochs,BATCH_SIZE=args.batch_size)
        if args.function == 'eval':
            model = args.pretrained_model_path + args.model
            if not os.path.isfile(model):
                print("No pretrained model found")
            else:
                evaluate(MODEL_PATH=model, DATA_DIR = args.data_path, BATCH_SIZE=args.batch_size*4)

