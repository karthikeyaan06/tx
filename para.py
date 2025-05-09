#pip install numpy tensorly
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac2
tensor = [tl.tensor(np.random.rand(4, 4)) for _ in range(3)]  # Example list of 2D matrices
rank = 2  # Number of components
weights, factors, projections = parafac2(tensor, rank)
print("Weights:", weights)
print("Factors:", factors)
print("Projections:", projections)
for i, factor in enumerate(factors):
    print(f"Factor {i} Shape:", factor.shape)
