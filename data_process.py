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
  
  def __getitem__(self,idx):
    """
    Returns I,M which are the fetched input and corresponding mask

    Keyword arguments:
      idx -- index used for the folderlist
    
    Returns:
      I_trans -- Input image transformed
      M_trans -- Corresponding mask transformed
    """

    folder_name = self.folders['0'].iloc[idx]

    I = read_image(os.path.join(self.image_dir, folder_name))[0,:,:]
    print(I.shape)
    F = fftshift(fftn(I))
    # Make a binary mask
    _,M = circ_sample.stitch_samples(F,3,50):
    F = np.stack([np.real(F),np.imag(F)],axis=0)
    
    # We want the mask channel number to equal the number of channels on F
    
 
    return torch.Tensor(I).to(torch.float32), torch.Tensor(M).to(torch.float32)
  
