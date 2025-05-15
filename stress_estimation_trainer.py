from models import StressDetectionModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stress_estimation_dataset import STRESS_dataset


epochs = 100

subjects = [i for i in range(1, 60) if i not in [1, 33, 49, 55]]
stress_dataset = STRESS_dataset(subjects=subjects, max_frame=3286)
train_dl = DataLoader(stress_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = StressDetectionModel(input_dim=stress_dataset[0][0].shape[1], embed_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

epoch_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_dl:  # batch_x: [B, T, F], batch_y: [B]
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
        
        logits, attn_weights = model(batch_x)
        loss = criterion(logits, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.shape[0]

    epoch_loss = running_loss / len(train_dl.dataset)
    epoch_losses.append(epoch_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')