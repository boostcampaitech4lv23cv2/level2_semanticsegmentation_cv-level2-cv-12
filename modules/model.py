from .models.Effiecient_unet import Efficient_UNet
from .models.FCN_resnet50 import FCN_ResNet50
from .models.Deconvnet_jj import DeconvNet_JJ
from .models.FCN_16s import FCN16s

_model_entrypoints = {
    "efficient_unet": Efficient_UNet,
    "fcn_resnet50": FCN_ResNet50,
    "deconvnet_jj": DeconvNet_JJ,
    "fcn_16s": FCN16s,
}

def model_entrypoint(model_name):
    return _model_entrypoints[model_name]

def is_exist_model(model_name):
    return model_name in _model_entrypoints

def create_model(model_name, **kwargs):
    if is_exist_model(model_name):
        create_model = model_entrypoint(model_name)
        model = create_model(**kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)
    
    return model