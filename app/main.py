from init import *
from config import *
from model_builder import BruitForceModelBuilder

# Load Config
config = Config()

if config.use_cuda:
    print("GPU available")
else:
    print("GPU not available")

# Build Models
modelBuilder = BruitForceModelBuilder()
models = modelBuilder.buildModel()




