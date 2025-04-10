import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import kagglehub
import string
import re
import os
from PIL import Image


# download the flickr8k data which is ~8k images with 5 captions each
path = kagglehub.dataset_download("adityajn105/flickr8k")
imgs_path = os.path.join(path, "Images")
captions_path = os.path.join(path, "captions.txt")


def clean_text(text):
    # Remove quotes (single and double) and punctuation
    text = text.replace('"', '').replace("'", '')
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# opens the captions file and loads it into image, caption pairs for easy loading
class Flickr8kDataset(Dataset):
    def __init__(self, images_dir, captions_file):
        """
        images_dir: directory with Flickr8k images
        captions_file: path to file containing image filenames and captions
        """

        self.images_dir = images_dir
        self.captions = []
        self.image_files = []
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),                     # resize image
            transforms.RandomHorizontalFlip(p=0.5),            # horiontal flip of image with probabiity of 50%
            transforms.ToTensor(),                             # convert to tensor [C x H x W]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # normalize to ImageNet mean/std
                                 std=[0.229, 0.224, 0.225])
        ])

        with open(captions_file, 'r') as file:
            next(file) # to lose the first line
            for line in file:
                cleaned_line = line.strip() # .strip() removes leading/trailing whitespace
                line_list = cleaned_line.split('.jpg,')
                img_name, caption = line_list[0]+".jpg", line_list[1]

                caption = clean_text(caption)
                # add beginning of sentence and end of sentence tokens
                tokens = ["<bos>"] + caption.lower().split() + ["<eos>"]
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)

                self.captions.append(tokens)
                self.image_files.append(img_name)

        # create reverse vocabulary to use when inferencing
        self.rev_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self.max_length = max(len(tokens) for tokens in self.captions)
        assert(len(self.captions) == len(self.image_files))


    def __len__(self):
        return len(self.captions)


    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        # Convert caption tokens to indices and pad
        token_list = self.captions[idx]
        cap_indices = [self.vocab.get(tok) for tok in token_list]
        # pad with 0 if less than max_length
        if len(cap_indices) < self.max_length:
            cap_indices += [0] * (self.max_length - len(cap_indices))

        caption_tensor = torch.tensor(cap_indices, dtype=torch.long)
        return image_tensor, caption_tensor
