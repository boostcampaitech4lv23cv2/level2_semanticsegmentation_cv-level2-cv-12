from .models.Effiecient_unet import Efficient_UNet
from .models.FCN_resnet50 import FCN_ResNet50
from .models.Deconvnet_jj import DeconvNet_JJ
from .models.FCN_16s import FCN16s
from .models.Efficientb0_unet_pp import Efficientb0_UNet_PP
from .models.Efficientb4_unet_pp import Efficientb4_UNet_PP
from .models.Efficientb4_deeplabv3p import Efficientb4_Deeplab_v3p
from .models.MVTb4_unet import MVTb4_UNet

_model_entrypoints = {
    "efficient_unet": Efficient_UNet,
    "fcn_resnet50": FCN_ResNet50,
    "deconvnet_jj": DeconvNet_JJ,
    "fcn_16s": FCN16s,
    "efficientb0_unet_pp": Efficientb0_UNet_PP,
    "efficientb4_unet_pp": Efficientb4_UNet_PP,
    "efficientb4_deeplabv3p": Efficientb4_Deeplab_v3p,
    "mvtb4_unet": MVTb4_UNet,
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