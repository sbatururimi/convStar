import lightning as L
from torch import nn
from convstar.multistages_convstar_net import MultistagesConvStarNet
from label_refinement.label_refinement_net import LabelRefinementNet

class HierarchicalConvRNN(L.LightningModule):
    def __init__(self, ms_convstar_net, label_refinment_net):
        super().__init__()
        self.ms_conv_star_net = ms_convstar_net
        self.label_refinment_net = label_refinment_net
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass