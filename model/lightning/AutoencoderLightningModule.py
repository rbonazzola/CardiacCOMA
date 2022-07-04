import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import Namespace

import numpy as np
from PIL import Image
import imageio

from typing import List, Mapping
from IPython import embed # uncomment for debugging
from model.Model3D import Autoencoder3DMesh as Autoencoder
# from data.synthetic.SyntheticMeshPopulation import SyntheticMeshPopulation
from image_helpers import *

losses_menu = {
  "l1": F.l1_loss,
  "mse": F.mse_loss
}

def mse(s1, s2=None):
    if s2 is None:
        s2 = torch.zeros_like(s1)
    return ((s1-s2)**2).sum(-1).mean(-1)


class AutoencoderLightning(pl.LightningModule):

    def __init__(self, model: Autoencoder, params: Namespace):

        """
        :param model: provide the PyTorch model.
        :param params: a Namespace with additional parameters
        """

        super(AutoencoderLightning, self).__init__()

        self.model = model
        self.params = params

        self.rec_loss = self._get_rec_loss()


    def _get_rec_loss(self):

        self.w_kl = self.params.loss.regularization.weight
        loss_type = self.params.loss.reconstruction.type.lower()
        return losses_menu[loss_type]


    def KL_div(self, mu, log_var):
        return -0.5 * torch.mean(torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)


    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)


    ########### COMMON BEHAVIOR
    def _unpack_data_from_batch(self, batch):
       
        #TODO: fix this 
        s = tuple([batch.get(k, None) for k in ["s"]])
        # z = torch.stack(z).transpose(0, 1).type_as(s_t)  # to get N_batches x latent_dim
        if len(s) == 1:
          return s[0]
        else:
          return s


    def _shared_step(self, batch, batch_idx):

        s = self._unpack_data_from_batch(batch)

        s_hat, bottleneck = self(s)
        # bottleneck, time_avg_shat, shat_t = self(s_t)

        if self.model._is_variational:
            mu, log_var = tuple([bottleneck[k] for k in ["mu", "log_var"]])
            kld_loss = self.KL_div(mu, log_var)
        else:
            kld_loss = torch.zeros_like(loss)

        recon_loss = self.rec_loss(s, s_hat)
        loss = recon_loss + self.w_kl * kld_loss

        loss_dict = {
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            "loss": loss
        }

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        return loss_dict


    def _average_over_batches(self, outputs: List[Mapping[str, torch.Tensor]], prefix: str = ""):

        keys = outputs[0].keys()
        loss_dict = {}
        for k in keys:
            avg_loss = torch.stack([x[k] for x in outputs]).mean().detach()
            loss_dict[prefix + k] = avg_loss
        return loss_dict


    ########### TRAINING
    def configure_optimizers(self):

        algorithm = self.params.optimizer.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.params.optimizer.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer


    def on_fit_start(self):

        #TODO: check of alternatives since .to(device) is not recommended
        #This is the most elegant way I found so far to transfer the tensors to the right device
        #(if this is run within __init__, I get self.device=="cpu" even when I use a GPU, so it doesn't work there)

        for matrix_type in ["upsample", "A_edge_index", "A_norm"]:
             for i, _ in enumerate(self.model.matrices["upsample"]):
                self.model.matrices[matrix_type][i] = self.model.matrices[matrix_type][i].to(self.device)


    def on_train_epoch_start(self):
        self.model.set_mode("training")

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)


    def training_epoch_end(self, outputs):

        # Aggregate metrics from each batch
        loss_dict = self._average_over_batches(outputs, prefix="training_")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)


    def _shared_eval_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)


    ########### VALIDATION
    def on_validation_start(self):
        self.model.set_mode("inference")


    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)


    def validation_epoch_end(self, outputs):
        loss_dict = self._average_over_batches(outputs, prefix="val_")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)


    ########### TESTING
    def on_test_start(self):
        self.model.set_mode("inference")


    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)


    def test_epoch_end(self, outputs):
        loss_dict = self._average_over_batches(outputs, prefix="test_")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)


    ########### PREDICTING
    def on_predict_start(self):
        self.model.set_mode("inference")


    def predict_step(self, batch, batch_idx):

        s = self._unpack_data_from_batch(batch)
        faces = self.model.template_mesh.f
        center_flag = self.params.dataset.preprocessing.center_around_mean
        s_hat, z = self(s)

        _s = s[0].cpu()
        _s_hat = s_hat[0].cpu() 

        if center:
            _s += self.model.template_mesh.v
            _s_hat += self.model.template_mesh.v
        
        png_prefix = f"mesh_{batch_idx}"

        orig_png, rec_png, full_png = ( f"{png_prefix}{suffix}.png" for suffix in ["_orig", "_rec", ""] )

        render_mesh_as_png(_s, faces, orig_png)
        render_mesh_as_png(_s_hat, faces, rec_png)

        merge_pngs_horizontally( orig_png, rec_png, full_png ) 

        self.logger.experiment.log_artifact(
            local_path=full_png,
            artifact_path="images", 
            run_id=self.logger.run_id
        )

        return {"z": z}

    def predict_epoch_end(self, outputs):

        z = torch.stack([x["z"] for x in outputs])






def merge_pngs_horizontally(png1, png2, output_png):
    # https://www.tutorialspoint.com/python_pillow/Python_pillow_merging_images.htm
    # Read the two images
    image1 = Image.open(png1)
    image2 = Image.open(png2)
    # resize, first image
    image1_size = image1.size
    # image2_size = image2.size
    new_image = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (250, 250, 250))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    new_image.save(output_png, "PNG")