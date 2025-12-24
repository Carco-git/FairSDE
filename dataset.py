import cv2
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image   

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean, std=std)
transform_train = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

resize = transforms.Resize((256, 256))
label_mapping = {'nv': 0, 'bkl': 0, 'mel': 1, 'akiec': 1, 'bcc': 0, 'df': 0, 'vasc': 0}
class HAM10000(Dataset):
    def __init__(self, sa='sex', meta_path='./data/ham/', img_path='./data/ham/',split='train'):
        
        data = pd.read_csv(os.path.join(meta_path, f'{split}.csv')) # split csv into train test val 8:1:1 first. Check the HAM10000_metadata.csv in the original dataset, split it into train test val 8:1:1 first.
        self.images = []
        self.labels = []
        self.sexes = []
        self.ages = []
        
        data = data[data['sex'].isin(['M', 'F'])]
        data['sex'] = data['sex'].replace({'M': 1, 'F': 0})
        data['label'] = data['dx'].map(label_mapping)
        self.meta = data
        
        self.meta['Age_multi'] = self.meta['age'].values.astype('int')
        self.meta['Age_multi'] = np.where(self.meta['Age_multi'].between(-1,19), 0, self.meta['Age_multi'])
        self.meta['Age_multi'] = np.where(self.meta['Age_multi'].between(20,39), 1, self.meta['Age_multi'])
        self.meta['Age_multi'] = np.where(self.meta['Age_multi'].between(40,59), 2, self.meta['Age_multi'])
        self.meta['Age_multi'] = np.where(self.meta['Age_multi'].between(60,79), 3, self.meta['Age_multi'])
        self.meta['Age_multi'] = np.where(self.meta['Age_multi']>=80, 4, self.meta['Age_multi'])

        for idx in tqdm(range(len(self.meta))):
            meta = self.meta.iloc[idx]
            img = Image.open(os.path.join(img_path, meta['Path']))
            img = img.convert('RGB')
            img = resize(img)
            label = meta['label']
            sex = int(meta['sex'])
            age = meta['Age_multi']
            self.images.append(img)
            self.labels.append(label)
            self.sexes.append(sex)
            self.ages.append(age)
        if(split=='train'):
            self.transform = transform_train
        else:    
            self.transform = transform_test
        self.sa = sa

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if(self.sa=='sex'):
            return self.transform(self.images[idx]), torch.tensor(self.labels[idx]), torch.tensor(self.sexes[idx])
        elif(self.sa=='age'):
            return self.transform(self.images[idx]), torch.tensor(self.labels[idx]), torch.tensor(int(self.ages[idx]==3 or self.ages[idx]==4) ) 
    
    
class mimic(Dataset):
    def __init__(self, meta_path='../data/', img_path='../data/{}', split='train', sensitive = 'race'):
        super()
        self.img_path = img_path.format(split)
        self.meta = pd.read_csv(os.path.join(meta_path, f'metadata_{split}_number.csv')) # use the processed metadata csv file from the preprocessing script, we used 50,000 samples from the original dataset
        self.images = []
        self.labels = []
        self.sexes = []
        self.isTrainset = False
        self.fixmatch = False
        self.transform = transform_test
        
        if(sensitive=='race'):
            self.sensitive_attributes = 2
        elif(sensitive=='sex'):
            self.sensitive_attributes = 2
        else:
            raise NotImplementedError
        
        for idx in (range(len(self.meta))):
            meta = self.meta.iloc[idx]
            label = meta['No Finding']
            if(sensitive=='race'):
                sensitive_att = int(meta['race']==0)
            elif(sensitive=='sex'):
                sensitive_att = meta['sex']
            else:
                raise NotImplementedError

            self.images.append(os.path.join(self.img_path, meta['path']))
            self.labels.append(label)
            self.sexes.append(sensitive_att)
            
        if(split=='train'):
            self.transform = transform_train
        else:    
            self.transform = transform_test

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = img.convert('RGB')
        return self.transform(img), self.labels[idx], self.sexes[idx]

from skimage.transform import resize as resize_eye
class EyeFair(Dataset): # copy the original dataset to data/eye
    def __init__(self, data_path=None, modality_type='rnflt', task='cls', resolution=224, need_shift=True, stretch=1.0,
                    depth=1, indices=None, attribute_type='race'):

        self.data_path = data_path
        self.modality_type = modality_type
        self.task = task
        self.attribute_type = attribute_type

        self.data_files = find_all_files(self.data_path, suffix='npz')
        if indices is not None:
            self.data_files = [self.data_files[i] for i in indices]

        self.race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}
        
        min_vals = []
        max_vals = []
        pos_count = 0
        min_ilm_vals = []
        max_ilm_vals = []
        self.raw_data = []
        self.race = []
        self.gender = []
        self.target = []
        for x in self.data_files:
            rnflt_file = os.path.join(self.data_path, x)
            raw_data = np.load(rnflt_file, allow_pickle=True)
            min_vals.append(raw_data['md'].astype(np.float32).item())
            max_vals.append(raw_data['md'].astype(np.float32).item())
            self.raw_data.append(raw_data['rnflt'])
            self.target.append(torch.tensor(float(raw_data['glaucoma'].item())))
            attr = 0
            cur_race = raw_data['race'].item()
            if cur_race in self.race_mapping:
                attr = self.race_mapping[cur_race]
            attr = torch.tensor(attr).long()
            self.race.append(attr)
            self.gender.append( torch.tensor(raw_data['male'].item()).long())
        self.normalize_vf = 30.0

        self.dataset_len = len(self.data_files)
        self.depth = depth
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        if self.modality_type == 'rnflt':
            rnflt_sample = self.raw_data[item]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize_eye(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)
        
        y = self.target[item]
        attr = 0
        if self.attribute_type == 'race':
            attr= self.race[item]
        elif self.attribute_type == 'gender':
            attr = self.gender[item]
        return data_sample, y.long(), attr
    


transform_train_celeba = transforms.Compose([
    transforms.RandomResizedCrop(size=64, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000)), # follow FSCL
])


transform_test_celeba = transforms.Compose([
    transforms.Resize(64), 
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000)), # follow FSCL
])

class CelebaDataset(Dataset):
    def __init__(self, root, split, ta = 31,  sa = 20,  transform=transform_test_celeba, download=True):
        self.root = root
        self.split = split
        self.transform = transform
        self.ta = ta
        self.sa = sa

        self.dataset = datasets.CelebA(root=self.root, split='all', download=download)
        self.train_indices, self.val_indices, self.test_indices = torch.load(os.path.join(self.root, 'celeba/split.pth'))
        if self.split == 'train':
            self.indices = self.train_indices
            self.transform = transform_train_celeba
        elif self.split == 'val':
            self.indices = self.val_indices
        else:
            self.indices = self.test_indices

        self.attr = self.dataset.attr[self.indices]
        

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        img, _ = self.dataset[index]
        ta = self.attr[idx][self.ta].item()
        sa = self.attr[idx][self.sa].item()
        if self.transform:
            img = self.transform(img)
        return img, ta, sa
        
        
class UTKDataset(Dataset):
    def __init__(self, root, split, ta , sa , transform=transform_test_celeba):
        self.root = root
        self.split = split
        self.transform = transform
        self.ta = ta
        self.sa = sa
        self.img_folder = os.path.join(self.root, 'UTKface_inthewild')
        
        train_filenames, val_filenames, test_filenames = torch.load(os.path.join(self.root, 'split.pth'))
        
        if self.split == 'train':
            self.file_list = train_filenames
            self.transform = transform_train_celeba
        elif self.split == 'val':
            self.file_list = val_filenames
        else:
            self.file_list = test_filenames

        self.ethnicity_list=[]
        self.gender_list=[]
        for i in range(len(self.file_list)):
            self.ethnicity_list.append(int(self.file_list[i].split('_')[2]=='0'))
            self.gender_list.append(int(self.file_list[i].split('_')[1]))
        
        
        if self.ta=='gender':
            self.num_classes=2
        elif self.ta=='ethnicity':
            self.num_classes=2
        else:
            raise NotImplementedError
        if self.sa=="gender":
            self.sensitive_attributes=2
        elif self.sa=="ethnicity":
            self.sensitive_attributes=2
        else:
            raise NotImplementedError

    def __getitem__(self, index1):
        gender=int(self.gender_list[index1])
        ethnicity=int(self.ethnicity_list[index1])
        ta=0
        sa=0

        img=Image.open( os.path.join(self.img_folder, self.file_list[index1]) ).convert('RGB') 

        if self.ta=='gender':
            ta=gender
        elif self.ta=='ethnicity':
            ta=ethnicity
        else:
            raise NotImplementedError
        
        if self.sa=="gender":
            sa=gender
        elif self.sa=="ethnicity":
            sa=ethnicity
        else:
            raise NotImplementedError
    
        if self.transform:
            img = self.transform(img)
            
        return img,ta,sa
    
    def __len__(self):
        return len(self.file_list)
    


def find_all_files(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files


        
        
import time
def load_dataset(args):
    if(args.dataset=='ham'):
        sensitive_attributes=2
        num_classes=2
        train_set, val_set, test_set = HAM10000(args.sa, split='train'), HAM10000(args.sa, split='val'), HAM10000(args.sa, split='test')

    elif(args.dataset=='mimiccxr'): # Licensed dataset, need to ask for permission to use
        train_set = mimic(meta_path=os.path.join(args.data_path,args.dataset), img_path=os.path.join(args.data_path,args.dataset), split='train', sensitive=args.sa)    
        val_set = mimic(meta_path=os.path.join(args.data_path,args.dataset), img_path=os.path.join(args.data_path,args.dataset), split='val', sensitive=args.sa)    
        test_set = mimic(meta_path=os.path.join(args.data_path,args.dataset), img_path=os.path.join(args.data_path,args.dataset), split='test', sensitive=args.sa)  
        sensitive_attributes=train_set.sensitive_attributes
        num_classes=2
    elif(args.dataset=='celeba'): # Use the given split, and download the dataset from their website
        sensitive_attributes=2
        num_classes=2
        train_set = CelebaDataset(root=os.path.join(args.data_path,''), ta=args.ta, split='train', download=True)
        val_set = CelebaDataset(root=os.path.join(args.data_path,''), ta=args.ta, split='val')
        test_set = CelebaDataset(root=os.path.join(args.data_path,''), ta=args.ta, split='test')

    elif(args.dataset=='utk'): # Use the given split, and download the dataset from their website
        if(args.sa=='race'):
            sa = 'ethnicity'
            ta = 'gender'
        else:
            ta = 'ethnicity'
            sa = 'gender'
        train_set = UTKDataset(root=os.path.join(args.data_path,args.dataset),ta=ta,sa=sa, split='train')
        val_set = UTKDataset(root=os.path.join(args.data_path,args.dataset),ta=ta,sa=sa, split='val')
        test_set = UTKDataset(root=os.path.join(args.data_path,args.dataset),ta=ta,sa=sa, split='test')
        sensitive_attributes=train_set.sensitive_attributes
        num_classes=train_set.num_classes

    elif(args.dataset=='HarvardGF'):  #Use their official split, and download the dataset from their website
        if(args.sa=='race'):
            sa = 'race'
            sensitive_attributes=3
        else:
            sa = 'gender'
            sensitive_attributes=2
        
        train_set = EyeFair(data_path=os.path.join(args.data_path,args.dataset,'Dataset/Training'), attribute_type=sa)  
        val_set = EyeFair(data_path=os.path.join(args.data_path,args.dataset,'Dataset/Validation'), attribute_type=sa)
        test_set = EyeFair(os.path.join(args.data_path,args.dataset,'Dataset/Test'), attribute_type=sa)
        num_classes=2
    
    return train_set, val_set, test_set, sensitive_attributes, num_classes
