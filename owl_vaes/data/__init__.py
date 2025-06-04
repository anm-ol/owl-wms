from . import mnist
from . import local_imagenet_256
from . import imagenet
from . import local_cod
def get_loader(data_id, batch_size):
    if data_id == "mnist":
        return mnist.get_loader(batch_size)
    if data_id == "local_imagenet_256":
        return local_imagenet_256.get_loader(batch_size)
    if data_id == "s3_imagenet":
        return imagenet.get_loader(batch_size)
    if data_id == "local_cod":
        return local_cod.get_loader(batch_size)