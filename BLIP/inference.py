import torch
import matplotlib.pyplot as plt
import numpy as np


def inference_val(model, val_loader, device, max_length):

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

            # Check if all sequences have ended with EOS
            if (next_token == 2).all():
                break

        return generated_tokens



