import torch
import matplotlib.pyplot as plt
import numpy as np


def inference_val(model, val_loader, device, max_length, reverse_vocab):

    for image, captions in val_loader:
        print(image.shape, captions.shape)
        # 0 bcz <bos> token
        generated_tokens = torch.tensor([0]*image.shape[0]).unsqueeze(1).to(device)

    with torch.no_grad():
        image = image.to(device)

        for _ in range(max_length):
            logits = model(image, generated_tokens, device)  # Shape: [batch_size, seq_len, vocab_size]
            # Get predicted token from the last timestep
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)  # Shape: [batch_size, 1]
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

    pred_captions = []
    for i in generated_tokens:
        # ignore <bos> and go till <eos>
        pred = [reverse_vocab[idx.item()] for idx in generated_tokens[i, 1:] if idx.item() != 2 ]
        pred_captions.append(pred)

    return pred_captions