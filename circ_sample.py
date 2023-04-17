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
  Mask = R <= rad

  return A*Mask



def stitch_samples(A,L,rad):
  # Generate a L x L patch of samples with no overlaps
  # Use the above circ_sample
  # Make sure L is odd

  samples = []
  steps = np.linspace(-(L//2),L//2,L,True).astype(int)
  for i in steps:
    for j in steps:
      sample = circ_sample(A,[i*rad*2,j*rad*2],rad)
      samples.append(sample)
  
  B = sum(samples)
  return B
