import lightning as L

# from torch import nn
# from convstar.multistages_convstar_net import MultistagesConvStarNet
# from label_refinement.label_refinement_net import LabelRefinementNet
import torch


class HierarchicalConvRNN(L.LightningModule):
    def __init__(
        self,
        ms_convstar_net,
        label_refinement_net,
        loss_weights_level1=None,
        loss_weights_level2=None,
        loss_weights_level3=None,
        lambda_1:float = 0.1,
        lambda_2: float = 0.3,
        lambda_3: float = 0.6,
        gamma: float = 0.6,
        grad_clip: float = 5
    ):
        super().__init__()
        self.save_hyperparameters()

        self.ms_conv_star_net = ms_convstar_net
        self.label_refinement_net = label_refinement_net

        self.loss_level_1 = torch.nn.CrossEntropyLoss(weight=loss_weights_level1)
        self.loss_level_2 = torch.nn.CrossEntropyLoss(weight=loss_weights_level2)
        self.loss_level_3 = torch.nn.CrossEntropyLoss(weight=loss_weights_level3)

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        self.gamma = gamma

        self.mean_loss_level_1 = 0.
        self.mean_loss_level_2 = 0.
        self.mean_loss_level_3 = 0. # most fine grained level

        self.mean_loss_refinement = 0.

        self.grad_clip = grad_clip

    def training_step(self, batch, batch_idx):
        input, target_glob, target_local_1, target_local_2 = batch

        # pass first through the multi-staged ConvStar network ------------------
        output_glob, output_local_1, output_local_2 = self.ms_conv_star_net(input)

        # compute losses per level
        l_level_1 = self.loss_level_1(output_local_1, target_local_1)
        l_level_2 = self.loss_level_2(output_local_2, target_local_2)
        l_level_3 = self.loss_level_3(output_glob, target_glob)

        total_loss = self.lambda_1 * l_level_1 + self.lambda_2 * l_level_2 + self.lambda_3 * l_level_3

        self.mean_loss_level_1 += l_level_1.data.cpu().numpy()
        self.mean_loss_level_2 += l_level_2.data.cpu().numpy()
        self.mean_loss_level_3 += l_level_3.data.cpu().numpy()

        # Label Refinement -------------------------------------------------
        output_glob_R = self.label_refinement_net([output_local_1, output_local_2, output_glob])
        refinement_loss = self.loss_level_3(output_glob_R, target_glob)
        self.mean_loss_refinement += refinement_loss.data.cpu().numpy()

        # combine losses, formula (6) in the paper
        total_loss = total_loss + self.gamma * refinement_loss

        return total_loss

    def on_after_backward(self):
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.ms_conv_star_net.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.label_refinement_net.parameters(), self.grad_clip)

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def parameters(self):
        return list(self.ms_conv_star_net.parameters()) + list(
            self.label_refinement_net.parameters()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer
