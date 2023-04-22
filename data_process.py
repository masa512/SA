import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import circ_sample
from scipy.fft import ifftn,fftn,fftshift,ifftshift


# Transport the dataset to the local disk
def transfer_data(in_path:str="", out_path:str="")->None:
  """}
  Transfer the data to the local - Also has loading bar for guestimate

  Keyword Argument:
  in_path (str) -- Input director of the data folder
  out_path (str) -- Output directory of the data folder

  Returns: None
  """

  # First listdir over the data folder to count number of folders in the directory (take out .csv)

  image_list = os.listdir(in_path)
  image_list = [s for s in image_list if not s.endswith('.csv')]

  # Delete folder if already exists
  if os.path.exists(out_path):
    shutil.rmtree(out_path)
  
  os.mkdir(out_path)

  # Run tqdm forloop
  for i in tqdm(range(len(image_list))):
    shutil.copytree(os.path.join(in_path,image_list[i]),os.path.join(out_path,image_list[i]))
  
  return None


# Define function for reading image
def read_image(image_path:str = ""):
    """
    Wrapper function to read image and raw image
    """


    I = plt.imread(os.path.join(image_path,'image.png'))

    return I
    
# Dataset definition

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class cell_dataset(Dataset):
  """
  Cell dataset class for fetch and process

  Class attributes:
  __init__ -- Constructor
  __len__ -- Returns number of examples in the dataset
  __getitem__ -- Fetches single example
  """

  def __init__(self, image_dir:str = '', df:pd.DataFrame = None, transform = None):
    """
    Constructor for dataset class 

    Keyword arguments:
      image_dir(str) -- base directory of the image
      df(pd.DataFrame) -- dataframe object containing folder names
      transform(transforms) -- transformations used for the dataset
    
    Returns : None
    """

    self.image_dir = image_dir
    self.folders = df
    self.transform = transform

  def __len__(self):
    """
    Returns length of the folder list
    """

    return len(self.folders)
  
  import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import circ_sample
from scipy.fft import ifftn,fftn,fftshift,ifftshift
import numpy as np
import torch

# Transport the dataset to the local disk
def transfer_data(in_path:str="", out_path:str="")->None:
  """}
  Transfer the data to the local - Also has loading bar for guestimate

  Keyword Argument:
  in_path (str) -- Input director of the data folder
  out_path (str) -- Output directory of the data folder

  Returns: None
  """

  # First listdir over the data folder to count number of folders in the directory (take out .csv)

  image_list = os.listdir(in_path)
  image_list = [s for s in image_list if not s.endswith('.csv')]

  # Delete folder if already exists
  if os.path.exists(out_path):
    shutil.rmtree(out_path)
  
  os.mkdir(out_path)

  # Run tqdm forloop
  for i in tqdm(range(len(image_list))):
    shutil.copytree(os.path.join(in_path,image_list[i]),os.path.join(out_path,image_list[i]))
  
  return None


# Define function for reading image
def read_image(image_path = ''):
    """
    Wrapper function to read image and raw image
    """
    I = plt.imread(image_path)
    # Rescale to 0-1
    I = I/255

    return I
    
# Dataset definition

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class dog_train_dataset(Dataset):

  def __init__(self,path = 'data/train',rad=30,L=5):
    self.path = path
    self.transforms = transforms
    self.file_list = [fn for fn in os.listdir(self.path) if fn.endswith('.jpg')]
    self.rad = rad
    self.L = L
  def __len__(self):
    return len(self.file_list)

  def __getitem__(self,idx):
      image_path = os.path.join(self.path,self.file_list[idx])
      I = np.mean(read_image(image_path),axis=-1,keepdims=False)
      # Take FFT
      F = fftshift(fftn(I))
      # Masking
      _,M = circ_sample.stitch_samples(F,self.L,self.rad)
      
      # All to torch tensor
      F = np.stack([np.real(F),np.imag(F)])
      F = torch.Tensor(F).float()
      M = torch.Tensor(M).float()
      I = torch.Tensor(I).float()

      return (I,F,M)




