import torch
from torch_geometric.datasets import GeometricShapes

dataset = GeometricShapes(root='data/GeometricShapes')
print(dataset)

data = dataset[0]
print(data)

import torch_geometric.transforms as T

dataset.transform = T.SamplePoints(num=256)

data = dataset[0]
print(data)

from torch_geometric.transforms import SamplePoints, KNNGraph

dataset.transform = T.Compose([SamplePoints(num=256), KNNGraph(k=6)])

data = dataset[0]
print(data)

from torch import Tensor
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import MessagePassing

from perforatedai import globalsFile as gf
from perforatedai import pb_models as PBM
from perforatedai import pb_utils as PBU

gf.switchMode = gf.doingHistory # When to switch between Dendrite learning and neuron learning. 
# How many normal epochs to wait for before switching modes.  
# Make sure this is higher than your scheduler's patience. 
gf.nEpochsToSwitch = 10  
gf.pEpochsToSwitch = 10  # Same as above for Dendrite epochs
gf.inputDimensions = [-1, 0, -1, -1] # The default shape of input tensors
gf.capAtN = True  # This ensures that Dendrite cycles do not take more epochs than neuron cycles
gf.device = 'cpu'
gf.initialCorrelationBatches = 3
gf.historyLookback = 1

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(
            Linear(in_channels + 3, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self,
        h_j: Tensor,
        pos_j: Tensor,
        pos_i: Tensor,
    ) -> Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)
    
from torch_geometric.nn import global_max_pool


class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, dataset.num_classes)

    def forward(self,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        # Perform two-layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # Classifier:
        return self.classifier(h)


model = PointNet()

from torch_geometric.loader import DataLoader

train_dataset = GeometricShapes(root='data/GeometricShapes', train=True)
train_dataset.transform = T.Compose([SamplePoints(num=256), KNNGraph(k=6)])
test_dataset = GeometricShapes(root='data/GeometricShapes', train=False)
test_dataset.transform = T.Compose([SamplePoints(num=256), KNNGraph(k=6)])

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

model = PointNet()
model = PBU.convertNetwork(model)
gf.pbTracker.initialize(
    doingPB = True, #This can be set to false if you want to do just normal training 
    saveName='GeometricPAI',  # Change the save name for different parameter runs
maximizingScore=True, # True for maximizing validation score, false for minimizing validation loss
makingGraphs=True)  # True if you want graphs to be saved

#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

gf.pbTracker.setOptimizer(torch.optim.Adam)
optimArgs = {'params':model.parameters(),'lr':0.01}
optimizer = gf.pbTracker.setupOptimizer(model, optimArgs)

criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    total_loss = 0
    total_correct = 0
    for data in train_loader:
        optimizer.zero_grad()
        logits = model(data.pos, data.edge_index, data.batch)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_loss / len(train_loader.dataset), total_correct / len(test_loader.dataset)


@torch.no_grad()
def test():
    model.eval()

    total_correct = 0
    for data in test_loader:
        logits = model(data.pos, data.edge_index, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(test_loader.dataset)


epoch = 0
while True:
    #Adding train_acc just for graphing purposes
    loss, train_acc = train()
    test_acc = test()
    gf.pbTracker.addTestScore(test_acc, 'Test Scores')
    # Adding training score as the valudation score because without validation the "best" network to test with would have to be picked based on the best training score.
    model, improved, restructured, trainingComplete = gf.pbTracker.addValidationScore(train_acc, 
    model,
    'GeometricPAI')
    if(trainingComplete):
        break
    elif(restructured): 
        optimArgs = {'params':model.parameters(),'lr':0.01}
        optimizer = gf.pbTracker.setupOptimizer(model, optimArgs)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    epoch += 1
print(f'Final: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
