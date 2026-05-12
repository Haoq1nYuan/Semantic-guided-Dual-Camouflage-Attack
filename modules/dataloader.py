from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, param):
        self.img_batch = param

    def __len__(self):
        return len(self.img_batch)

    def __getitem__(self, idx):
        return self.img_batch[idx]

def UpdateDataloader(img_batch, batch_size):
    dataset = ImageDataset(img_batch)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader