import lightning as L
# from torch import nn
# from convstar.multistages_convstar_net import MultistagesConvStarNet
# from label_refinement.label_refinement_net import LabelRefinementNet
import torch

class HierarchicalConvRNN(L.LightningModule):
    def __init__(self, ms_convstar_net, label_refinement_net):
        super().__init__()
        self.ms_conv_star_net = ms_convstar_net
        self.label_refinement_net = label_refinement_net
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def parameters(self):
        return list(self.ms_conv_star_net.parameters()) + list(self.label_refinement_net.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)