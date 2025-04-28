from torch.utils.data import Dataset
from torchvision import transforms
import torch

class VQADataset(Dataset):

    def __init__(self, ds, transform=None):
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


        for i, question_answer in enumerate(self.ds):
          question = question_answer["question"][:-1]
          answer = question_answer["multiple_choice_answer"]
          image = question_answer["image"]

          self.questions.append(self.encode(question))
          self.answers.append(self.encode(answer))
          # so i dont have to store img in ram
          self.images.append(i)

        self.rev_vocab = {idx: tok for tok, idx in self.vocab.items()}


    def encode(self, text):
        tokens = ["<bos>"] + text.lower().strip().split() + ["<eos>"]
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

        indices = [self.vocab.get(tok) for tok in tokens]
        return torch.tensor(indices, dtype=torch.long)


    def __len__(self):
        return len(self.questions)


    def __getitem__(self, idx):
        image_index = self.images[idx]
        image_tensor = self.transform(self.ds[image_index]['image'])

        return image_tensor, self.questions[idx], self.answers[idx]



def collate_fn(batch):
    images = [i[0] for i in batch]
    images_tensor = torch.stack(images, axis = 0)

    questions = [i[1] for i in batch]
    answers = [i[2] for i in batch]
    padded_qns = torch.nn.utils.rnn.pad_sequence(questions, batch_first=True, padding_value=0)
    padded_ans = torch.nn.utils.rnn.pad_sequence(answers, batch_first=True, padding_value=0)

    return images_tensor, padded_qns, padded_ans