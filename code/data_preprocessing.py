from torchvision import transforms,datasets
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np


def data_preprocess(train_datadir, batch_size):
    data_transforms = transforms.Compose([
        transforms.Resize([64,64]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    Data = datasets.ImageFolder(train_datadir, transform=data_transforms)

    valid_size = 0.2
    num_train = len(Data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, eval_idx = indices[split:], indices[:split]

    train_dataset = SubsetRandomSampler(train_idx)
    eval_dataset = SubsetRandomSampler(eval_idx)

    train_dataloader = DataLoader(Data, sampler = train_dataset, batch_size = batch_size)
    eval_dataloader = DataLoader(Data, sampler = eval_dataset, batch_size = batch_size)

    return train_dataloader,eval_dataloader
