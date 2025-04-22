from torch.utils.data import Dataset
from torchvision import transforms
import torch


class vqaDataset(Dataset):

    def __init__(self, ds, transform=None, max_length=None):
        """
        transform: image transformations (resize, normalize, etc.)
        max_length: max caption length (including <bos> and <eos>) for padding
        """
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),                     # resize image
            transforms.ToTensor(),                             # convert to tensor [C x H x W]
            transforms.Lambda(lambda image: image.repeat(3, 1, 1) if image.shape[0] == 1 else image),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # normalize to ImageNet mean/std
                                 std=[0.229, 0.224, 0.225])
        ])

        self.questions = []
        self.answers = []
        self.images = []
        self.all_tokens = []

        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.ds = ds["validation"]


        for i, question_answer in enumerate(ds):
          question = question_answer["question"][:-1]
          answer = question_answer["multiple_choice_answer"]

          self.questions.append(self.tokenize(question))
          self.answers.append(self.tokenize(answer))
          self.images.append(i)

        self.rev_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self.max_length = max(len(tokens) for tokens in self.questions)


    def tokenize(self, text):
        tokens = ["<bos>"] + text.lower().strip().split() + ["<eos>"]
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        return tokens


    def __len__(self):
        return len(self.questions)


    def encode_and_pad(self, tokens):
        indices = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]

        # pad with 0 if less than max_length
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)


    def __getitem__(self, idx):
        image_index = self.images[idx]
        image_tensor = self.transform(self.ds['validation'][image_index]['image'])

        question_tensor = self.encode_and_pad(self.questions[idx])
        answer_tensor = self.encode_and_pad(self.answers[idx])

        return image_tensor, question_tensor, answer_tensor