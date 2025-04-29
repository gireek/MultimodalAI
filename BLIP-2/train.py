from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch import nn
from datasets import load_dataset
from dataset_collate import VQADataset, collate_fn
from torch.utils.data import random_split
from model import blip_vqa
from torch import optim
from torch.cuda.amp import autocast, GradScaler



ds = load_dataset("merve/vqav2-small")
data = VQADataset(ds)
train_size = int(0.97 * len(data))
val_size = len(data) - train_size

train_dataset, val_dataset = random_split(data, [train_size, val_size])
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

max_qn_len = max([len(q) for q in data.questions])
max_ans_len = max([len(a) for a in data.answers])
vocab_size = len(data.vocab)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = blip_vqa(vocab_size, max_qn_len, max_ans_len).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scaler = GradScaler()  # for mixed precision
num_epochs = 10
grad_accum_steps = 1  # set >1 if you need to accumulate gradients

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, questions, answers) in enumerate(train_loader):
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)

        answers_in = answers[:, :-1]   # decoder input
        answers_out = answers[:, 1:]   # labels

        with autocast(): # Forward pass with mixed precision
            logits = model(images, questions, answers_in, device)  # [batch, seq_len-1, vocab_size]
            loss = criterion(logits.view(-1, logits.size(-1)), answers_out.reshape(-1))

        # Backpropagation
        scaler.scale(loss).backward()
        # Gradient accumulation: if we've done accum_steps forward passes, or if last batch, step optimizer
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item()

        if (batch_idx) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} - Avg Loss: {avg_loss:.4f}")
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    with torch.no_grad():
        model.eval()
        total_val_loss = 0.0

        for batch_idx, (images, questions, answers) in enumerate(val_loader):
          images = images.to(device)
          questions = questions.to(device)
          answers = answers.to(device)

          answers_in = answers[:, :-1]
          answers_out = answers[:, 1:] 

          with autocast(): # Forward pass with mixed precision
              logits = model(images, questions, answers_in, device)  # [batch, seq_len-1, vocab_size]
              loss = criterion(logits.view(-1, logits.size(-1)), answers_out.reshape(-1))

          total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")