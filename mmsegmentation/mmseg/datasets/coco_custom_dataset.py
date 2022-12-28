from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOCustomDataset(CustomDataset):
    CLASSES = [
        'Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
        'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
    ]

    PALLETE = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
            [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
            [64, 64, 128], [128, 0, 192]]

    def __init__(self, **kwargs):
        super(COCOCustomDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)