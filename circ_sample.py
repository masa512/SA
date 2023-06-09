import numpy as np

def circ_sample(A,loc,rad):
  # Samples circular region from matrix A at location loc and with radius rad
  # Evaluate radius 

  N,M = A.shape
  x = np.arange(M) - M//2
  y = np.arange(N) - N//2

  X,Y = np.meshgrid(x,y)

  # Radius matrix
  R = np.sqrt((X-loc[1])**2+(Y-loc[0])**2)


  # Return the subsection
  Mask = 1*(R <= rad)

  return A*Mask,Mask



def stitch_samples(A,L,rad):
  # Generate a L x L patch of samples with no overlaps
  # Use the above circ_sample
  # Make sure L is odd

  samples = []
  masks = []
  steps = np.linspace(-(L//2),L//2,L,True).astype(int)
  for i in steps:
    for j in steps:
      sample,mask = circ_sample(A,[i*rad*2,j*rad*2],rad)
      samples.append(sample)
      masks.append(mask)

  
  B = sum(samples)
  M = sum(masks)

  M[M>0] = 1

  return B,M
