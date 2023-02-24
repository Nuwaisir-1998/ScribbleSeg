import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
import numpy as np
import pandas as pd
import torch.nn.init
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
import math
from scipy import spatial
import json
import random


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False