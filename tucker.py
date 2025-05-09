#pip install numpy scipy tensorly scikit-learn
import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
tensor = tl.tensor(np.random.rand(4, 4, 4))  # Example 3D tensor
ranks = [2, 2, 2]  # Rank for Tucker decomposition
core, factors = tucker(tensor, ranks)
print("Core Tensor:", core)
print("Factors:", factors)
print("Core Shape:", core.shape)
for i, factor in enumerate(factors):
    print(f"Factor {i} Shape:", factor.shape)
