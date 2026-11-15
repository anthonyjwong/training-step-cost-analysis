import sys

import torch
import torchvision

if "--config" in sys.argv:
    print(torch.__config__.show())
else:
    print("torch", torch.__version__, "torchvision", torchvision.__version__)
