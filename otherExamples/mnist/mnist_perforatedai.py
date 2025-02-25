from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM
from perforatedai import pb_utils as PBU

class Net(nn.Module):
    def __init__(self, num_classes, width):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, int(32*width), 3, 1)
        self.conv2 = nn.Conv2d(int(32*width), int(64*width), 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(int(9216*width), int(128*width))
        self.fc2 = nn.Linear(int(128*width), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0

    #Loop over all the batches in the dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        #Pass the data through your model to get the output
        output = model(data)
        #Calculate the error
        loss = F.nll_loss(output, target)
        #Backpropagate the error through the network
        loss.backward()
        if(args.dataParallel):
            model.gatherData()
        #Modify the weights based on the calculated gradient
        optimizer.step()
        #Display Metrics
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        #Determine the predictions the network was making
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #Increment how many times it was correct
        correct += pred.eq(target.view_as(pred)).sum()
    #Add the new score to the tracker which may restructured the model with PB Nodes
    PBG.pbTracker.addExtraScore(100. * correct / len(train_loader.dataset), 'train') 
    model.to(device)


def test(model, device, test_loader, optimizer, scheduler, args):
    model.eval()
    test_loss = 0
    correct = 0
    #Dont calculate Gradients
    with torch.no_grad():
        #Loop over all the test data
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #Pass the data through your model to get the output
            output = model(data)
            #Calculate the error
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #Determine the predictions the network was making
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #Increment how many times it was correct
            correct += pred.eq(target.view_as(pred)).sum()

    #Display Metrics
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if(args.dataParallel):
        #Add the new score to the tracker which may restructured the model with PB Nodes
        model, improved, restructured = PBG.pbTracker.addValidationScore(100. * correct / len(test_loader.dataset), 
        model.module,
        args.save_name) 
        model = PBM.PAIDataParallel(model, device_ids=range(torch.cuda.device_count())).to(PBG.device)
    else:
        #Add the new score to the tracker which may restructured the model with PB Nodes
        model, improved, restructured, trainingComplete = PBG.pbTracker.addValidationScore(100. * correct / len(test_loader.dataset), 
        model,
        args.save_name) 
        model.to(device)
    #If it was restructured reset the optimizer and scheduler
    if(restructured): 
        optimArgs = {'params':model.parameters(),'lr':args.lr}
        schedArgs = {'step_size':1, 'gamma': args.gamma}
        optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimArgs, schedArgs)

    return model, optimizer, scheduler, trainingComplete


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--save-name', type=str, default='PB')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--width', type=float, default=1.0, metavar='M',
                        help='width multiplier')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataParallel', type=int, default=0,
                        help='using data parallel with multi gpus')
    parser.add_argument('--pretrained', type=str, default='',
                        help='path to a pretrained model')
    parser.add_argument('--test-head', type=int, default=0,
                        help='if true only test converting the head of the network')
    parser.add_argument('--test-backbone', type=int, default=0,
                        help='if true only test converting the backbone of the network')
    parser.add_argument('--doingPB', type=int, default=1,
                        help='if false dont convert model')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if(args.dataset == 'MNIST'):
        num_classes = 10
        #Define the data loaders
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset1 = datasets.MNIST('./data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('./data', train=False,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    elif(args.dataset == 'EMNIST'):
        num_classes = 47
        transform_train = transforms.Compose(
                [ 
                    transforms.CenterCrop(26),
                    transforms.Resize((28,28)),
                    transforms.RandomRotation(10),      
                    transforms.RandomAffine(5),
                    transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ])
        transform_test = transforms.Compose(
                [ transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
                ])
        #Dataset
        dataset1 = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform_train)

        dataset2 = datasets.EMNIST(root='./data',  split='balanced', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #Set up some global parameters for PAI code
    PBG.switchMode = PBG.doingHistory # This is when to switch between PAI and regular learning
    #PBG.retainAllPB = True
    PBG.inputDimensions = [-1, 0, -1, -1] #this is the shape of inputs, for a standard conv net this will work
    PBG.nEpochsToSwitch = 25  #This is how many normal epochs to wait for before switching modes.  Make sure this is higher than your schedulers patience. 
    PBG.pEpochsToSwitch = 25  #Same as above for PAI epochs
    PBG.capAtN = True #Makes sure subsequent rounds last max as long as first round
    PBG.initialHistoryAfterSwitches = 5
    PBG.testSaves = True

    #Create the model
    if(args.pretrained == ''):
        model = Net(num_classes, args.width)
    else:
        model = torch.load(args.pretrained)

    if(args.dataParallel):
        PB.newDataParallel = True
    
    #Convert the network to be a PAI Network
    if(args.test_head):
        model.fc2 = PBU.convertNetwork(model.fc2, layerName = 'layer2')
    if(args.test_backbone):
        model.conv1 = PBU.convertNetwork(model.conv1, layerName = 'layer2')
        model.conv2 = PBU.convertNetwork(model.conv2, layerName = 'layer2')
        model.fc1 = PBU.convertNetwork(model.fc1, layerName = 'layer2')
    else:
        model = PBU.convertNetwork(model)
            #Setup a few extra parameters
    PBG.pbTracker.initialize(
        doingPB = args.doingPB, #This can be set to false if you want to do just normal training 
        saveName=args.save_name,  # change the save name for different parameter runs
        maximizingScore=True, #true for maximizing score, false for reducing error
        makingGraphs=True)  #true if you want graphs to be saved

    
    if(args.dataParallel):
        model = PBM.PAIDataParallel(model, device_ids=range(torch.cuda.device_count())).to(PBG.device)
    print('Running with %d devices' % (torch.cuda.device_count()))



    model = model.to(device)
    
    #Setup the optimizer and scheduler
    PBG.pbTracker.setOptimizer(optim.Adadelta)
    PBG.pbTracker.setScheduler(StepLR)
    optimArgs = {'params':model.parameters(),'lr':args.lr}
    schedArgs = {'step_size':1, 'gamma': args.gamma}
    optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimArgs, schedArgs)


    #Run your epochs of training and testing
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        model, optimizer, scheduler, trainingComplete = test(model, device, test_loader, optimizer, scheduler, args)
        if(trainingComplete):
            break

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
