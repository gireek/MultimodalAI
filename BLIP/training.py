from dataset import Flickr8kDataset
from model import ImageCaptioningModel
import kagglehub
import torch
import torch.nn as nn
import os
from torch import optim
from torch.cuda.amp import autocast, GradScaler



# download the flickr8k data which is ~8k images with 5 captions each
path = kagglehub.dataset_download("adityajn105/flickr8k")
imgs_path = os.path.join(path, "Images")
captions_path = os.path.join(path, "captions.txt")


data = Flickr8kDataset(imgs_path, captions_path)
max_length = data.max_length
vocab_size = len(set(data.vocab))
print(vocab_size, max_length)


dataset_size = len(data)
train_size = 37000
val_size = dataset_size - train_size

train_dataset = torch.utils.data.Subset(data, range(train_size))
val_dataset = torch.utils.data.Subset(data, range(train_size, dataset_size))
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageCaptioningModel(vocab_size, max_length).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  # for mixed precision
num_epochs = 15
grad_accum_steps = 1  # set >1 if you need to accumulate gradients

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)

        decoder_input = captions[:, :-1]
        target = captions[:, 1:]

        with autocast(): # Forward pass with mixed precision
            logits = model(images, decoder_input, device)  # [batch, seq_len-1, vocab_size]
            loss = criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))

        # Backpropagation
        scaler.scale(loss).backward()
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item()

        if (batch_idx) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} - Avg Loss: {avg_loss:.4f}")
    # End of epoch: compute average loss
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    # similarly for val data
    with torch.no_grad():
        model.eval()
        total_val_loss = 0.0
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)

            decoder_input = captions[:, :-1]
            target = captions[:, 1:]

            logits = model(images, decoder_input, device)
            loss = criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

