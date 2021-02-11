import argparse
import os
import utils



parser = argparse.ArgumentParser(description='Train a new network',)
parser.add_argument('data_dir', action='store')
parser.add_argument('--save-dir', action='store', default= os.getcwd())
parser.add_argument('--arch', action='store', default='vgg13')
parser.add_argument('--learning_rate', action='store', default=0.002, type=float)
parser.add_argument('--gpu', action='store', default=True)
parser.add_argument('--hidden_units', action='store', default=512, type=int)
parser.add_argument('--epochs', action='store', default = 5, type=int)
parser.add_argument('--checkpoint', action='store')

args = parser.parse_args()
print(args)

train_dataset, train_dataloader = utils.load_train_data(args.data_dir + "/train/")
valid_dataset, valid_dataloader = utils.load_valid_data(args.data_dir + "/valid/")
test_dataset, test_dataloader = utils.load_test_data(args.data_dir + "/test/")

model = utils.create_network(args.arch, args.hidden_units, True)
criterion = utils.create_criterion()
optimizer = utils.create_optimizer(model, args.learning_rate)

model = utils.train_model(model, criterion, optimizer, train_dataloader, valid_dataloader,args.epochs, args.gpu)

utils.save_checkpoint(model, args.arch, args.hidden_units, train_dataset, args.save_dir)

