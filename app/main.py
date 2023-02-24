from init import *
from config import *


config = Config()

if config.use_cuda:
    print("GPU available")
else:
    print("GPU not available")



