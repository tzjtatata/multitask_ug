from .cifar import (
    CIFARDataModule
)
from .NYUv2 import (
    NYUv2DataModule,
    NYUv2SAInterpDataModule,
    NYUv2STAGapDataModule
)
from .cityscapes import (
    CityscapesDataModule,
    CityscapesSAInterpDataModule,
    CityscapesSAITAPlusDataModule,
    CityscapesSADataModule,
    CityscapesMTMODataModule
)
from .nyud import (
    NYUMTDataModule
)
from .nyud_labelmap import (
    NYULMDataModule
)
from .celebA import (
    CelebADataModule
)
from .loren_datasets import LorenCityscapes
