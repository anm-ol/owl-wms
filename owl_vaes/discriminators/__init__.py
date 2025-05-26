from .res import R3GANDiscriminator

def get_discrminator_cls(model_id):
    if "model_id" == "r3gan":
        return R3GANDiscriminator
