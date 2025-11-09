import os
import glob
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

class DebrisDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg")) + glob.glob(os.path.join(data_dir, "*.png"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, os.path.basename(image_path)

class DataLoader:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
    
    def create_dataloader(self, data_dir, transform=None):
        dataset = DebrisDataset(data_dir, transform)
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=True)