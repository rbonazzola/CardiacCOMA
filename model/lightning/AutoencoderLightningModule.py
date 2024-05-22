import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv
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

N_IMAGES_LOGGED = 10
CAMERA_VIEWS = ['xz', 'xy', 'yz']

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

        self.optimizer = self.configure_optimizers()
        
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
        id, s = tuple([batch.get(k, None) for k in ["id", "s"]])
        # z = torch.stack(z).transpose(0, 1).type_as(s_t)  # to get N_batches x latent_dim
        return id, s
        #if len(s) == 1:
        #  return s[0]
        #else:
        #  return id, s


    def _shared_step(self, batch, batch_idx):

        id, s = self._unpack_data_from_batch(batch)

        s_hat, bottleneck = self(s)
        # bottleneck, time_avg_shat, shat_t = self(s_t)

        recon_loss = self.rec_loss(s, s_hat)

        if self.model._is_variational:
            mu, log_var = tuple([bottleneck[k] for k in ["mu", "log_var"]])
            kld_loss = self.KL_div(mu, log_var)
        else:
            kld_loss = torch.zeros_like(recon_loss)

        loss = recon_loss + self.w_kl * kld_loss

        loss_dict = {
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
            "loss": loss
        }

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        loss_dict["id"] = id
        return loss_dict


    def _average_losses_over_batches(self, outputs: List[Mapping[str, torch.Tensor]], prefix: str = ""):

        keys = [x for x in outputs[0].keys() if x.endswith("loss")]
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
        self.optimizer = algorithm(self.model.parameters(), **parameters)
        return self.optimizer


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
        loss_dict = self._average_losses_over_batches(outputs, prefix="training_")
        self._collect_ids(outputs, "training_ids.txt")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)


    def _shared_eval_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)


    ########### VALIDATION
    def on_validation_start(self):
        self.model.set_mode("inference")


    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)


    def validation_epoch_end(self, outputs):
        loss_dict = self._average_losses_over_batches(outputs, prefix="val_")
        self._collect_ids(outputs, "validation_ids.txt")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)


    ########### TESTING
    def on_test_start(self):
        self.model.set_mode("inference")


    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)


    def test_epoch_end(self, outputs):
        loss_dict = self._average_losses_over_batches(outputs, prefix="test_")
        self._collect_ids(outputs, "test_ids.txt")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)


    ########### PREDICTING
    def on_predict_start(self):
        self.model.set_mode("inference")


    def predict_step(self, batch, batch_idx):

        id, s = self._unpack_data_from_batch(batch)
        # faces = self.model.template_mesh.f
        # center_flag = self.params.dataset.preprocessing.center_around_mean        
        s_hat, z = self(s)
        # print(s.mean(0))
        # print(s_hat.mean(0))
        
        z = z['mu']
        _mse = mse(s, s_hat)
        print(_mse.mean())
        # _s = s[0].cpu()
        # _s_hat = s_hat[0].cpu() 

        # if batch_idx < N_IMAGES_LOGGED:
# 
        #     if center_flag:
        #         _s += self.model.template_mesh.v
        #         _s_hat += self.model.template_mesh.v
        # 
        #     png_prefix = f"mesh_{batch_idx}"
  # 
        #     orig_and_rec_pngs = []
        #     for camera in CAMERA_VIEWS:
        #         orig_png, rec_png, orig_and_rec_png = [ f"{png_prefix}_{camera}_{suffix}.png" for suffix in ["_orig", "_rec", ""] ]
        #         render_mesh_as_png(_s, faces, orig_png, camera_position=camera)
        #         render_mesh_as_png(_s_hat, faces, rec_png, camera_position=camera)
        #         merge_pngs( [orig_png, rec_png], orig_and_rec_png, how="horizontally") 
        #         orig_and_rec_pngs.append(orig_and_rec_png)
# 
        #     full_png = png_prefix + ".png"
        #     merge_pngs(orig_and_rec_pngs, full_png, how="vertically")
# 
        #     self.logger.experiment.log_artifact(
        #         local_path=full_png,
        #         artifact_path="images", 
        #         run_id=self.logger.run_id
        #     )
# 
        return {"id": id, "z": z, "mse": _mse}

    def on_predict_epoch_end(self, outputs):

        z_filename = self.trainer.datamodule._z_filename
        mse_filename = self.trainer.datamodule._mse_filename
        
        z_df = self._log_z_vectors(outputs, z_filename)
        mse_df = self._log_mse(outputs, mse_filename)
        
        print(z_df.head())
        print(mse_df.head())

        #perf_file = "{}/performance.csv".format(output_dir)
        #perf_df = pd.DataFrame(None, columns=["mse"])

    def _collect_ids(self, outputs, filename=None):
        ids = [x["id"] for x in outputs]
        ids = [id for sublist in ids for id in sublist]
        ids = pd.DataFrame(ids, columns=["ID"])
        if filename is not None:
            ids.to_csv(filename, index=False)
            self.logger.experiment.log_artifact(
                local_path = filename,
                artifact_path = "output", run_id=self.logger.run_id
            )
        return ids

    
    def _log_mse(self, outputs, filename):
        
        ids = self._collect_ids(outputs[0])
        _mse = torch.concat([x["mse"] for x in outputs[0]], axis=0)
        # _mse = [x for x in _mse.cpu().numpy()].transpose()
        _mse = _mse.cpu().numpy().transpose()
                
        # mse_df = pd.DataFrame([ids, _mse]).transpose()
        mse_df = pd.DataFrame([_mse]).transpose()
        mse_df = pd.concat([ids, mse_df], axis=1)
        mse_df.columns = ["ID", "MSE"]
        mse_df.to_csv(filename, index=False)
        
        self.logger.experiment.log_artifact(
            local_path = filename,
            artifact_path = "output", run_id=self.logger.run_id
        )
        
        return mse_df
    
    
    def _log_z_vectors(self, outputs, filename):
        
        z = torch.concat([x["z"] for x in outputs[0]], axis=0)
        z_columns = [f"z{i:03d}" for i in range(z.shape[1])] # z001, z002, z003, ...
        z_df = pd.DataFrame(np.array(z), columns=z_columns)
        
        ids = self._collect_ids(outputs[0])
        z_df = pd.concat([ids, z_df], axis=1)
        z_df.to_csv(filename, index=False)       
 
        corr_matrix = z_df.corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.savefig("z_corr_matrix.png")

        self.logger.experiment.log_artifact(
            local_path = "z_corr_matrix.png",
            artifact_path = "output", run_id=self.logger.run_id
        )

        self.logger.experiment.log_artifact(
            local_path = filename,
            artifact_path = "output", run_id=self.logger.run_id
        )
        
        return z_df


