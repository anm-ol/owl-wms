from . import mnist
from . import local_imagenet_256
def get_loader(data_id, batch_size):
    if data_id == "mnist":
        return mnist.get_loader(batch_size)
    if data_id == "local_imagenet_256":
        return local_imagenet_256.get_loader(batch_size)
