from models import StressDetectionModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from stress_estimation_dataset import STRESS_dataset
import numpy as np
import random


epochs = 100
subjects = [i for i in range(1, 60) if i not in [1, 33, 49, 55]]  # Exclude subjects 1, 33, 49, and 55.
tasks = range(1, 12)  # Tasks are numbered from 1 to 11.
last_validations = []
min_validations = {}
# Create a list with tuples of all possible combinations of subjects and tasks
subject_task_pairs = [(s, t) for s in subjects for t in tasks]

for i in range(100):  
    # Use 3 pairs for validation.
    valid_pairs = random.sample(subject_task_pairs, 3)
    print(f"Validation pairs: {valid_pairs}")
    train_pairs = [p for p in subject_task_pairs if p not in valid_pairs]
    train_stress_dataset = STRESS_dataset(sequence_pairs=train_pairs)
    valid_stress_dataset = STRESS_dataset(sequence_pairs=valid_pairs)
    train_dl = DataLoader(train_stress_dataset, batch_size=16, shuffle=True)
    valid_dl = DataLoader(valid_stress_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = StressDetectionModel(input_dim=train_stress_dataset[0][0].shape[1], embed_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    training_losses = []
    validation_losses = []

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

        epoch_training_loss = running_loss / len(train_dl.dataset)
        training_losses.append(epoch_training_loss)

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for batch_x, batch_y in valid_dl:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                
                valid_loss += loss.item() * batch_x.shape[0]

            epoch_validation_loss = valid_loss / len(valid_dl.dataset)
            validation_losses.append(epoch_validation_loss)

    print(f'Fold [{i+1}/100], Validation Loss: {epoch_validation_loss:.4f}, Best Validation Loss: {min(validation_losses):.4f}')
    last_validations.append(epoch_validation_loss)
    min_validations[min(validation_losses)] = validation_losses.index(min(validation_losses))
print(last_validations)
print(min_validations)
print("-----------------------")
print(np.mean(last_validations))
print(np.mean(list(min_validations.keys())))