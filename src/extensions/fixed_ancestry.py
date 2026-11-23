from torch.utils.data import Dataset
import torch
from torchvision import transforms as T
from PIL import Image
class FixedAncestryDataset(Dataset):
    """A deterministic, offline 384px augmentation pool; no test originals inserted."""
    def __init__(self, frame, images):
        self.frame = frame.reset_index(drop=True); self.images = images
        self.transform = T.Compose([T.Resize((384,384)), T.ToTensor(),
                                    T.Normalize([.485,.456,.406], [.229,.224,.225])])
        self.augment = T.Compose([T.RandomRotation(30),
            T.RandomAffine(degrees=0, translate=(.1,.1), scale=(.8,1.2), shear=15),
            T.ColorJitter(brightness=.15, contrast=.2, saturation=.15, hue=.1), T.Resize((384,384))])

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        r = self.frame.iloc[index]
        image = Image.open(self.images[r.filename]).convert('RGB')
        if int(r.variant) > 0:
            # torchvision PIL transforms draw from torch; avoid changing the loader RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(r.augmentation_seed)); image = self.augment(image)
        return self.transform(image), int(r.label), r.patient_cluster


