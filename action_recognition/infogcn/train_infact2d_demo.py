import sys
import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from feeders.feeder_coco_2d import Feeder

dataset = Feeder(
    data_path='../Data/InfActPrimitive/2d/InfAct_plus.pkl',
    split='train'
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

C, T, V, M = 3, 100, 17, 1
num_class = len(set(dataset.label))
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(C*T*V*M, num_class)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for i, (data, label, idx) in enumerate(loader):
    data = data.float()
    label = label.long()

    out = model(data)
    loss = criterion(out, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Step {i}: loss={loss.item():.4f}")

    if i > 30:
        break