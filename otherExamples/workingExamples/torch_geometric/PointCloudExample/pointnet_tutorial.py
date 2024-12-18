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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        logits = model(data.pos, data.edge_index, data.batch)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test():
    model.eval()

    total_correct = 0
    for data in test_loader:
        logits = model(data.pos, data.edge_index, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(test_loader.dataset)

for epoch in range(1, 51):
    loss = train()
    test_acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')
