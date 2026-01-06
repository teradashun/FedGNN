from torch_geometric.datasets import Planetoid
from sklearn.cluster import k_means
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from torch_geometric.data import Data

dataset = Planetoid(root="./Cora", name="Cora")
data = Data(dataset[0].x, dataset[0].edge_index, dataset[0].y)
print(data.x)
