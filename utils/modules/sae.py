import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dict_size = int(self.cfg["dict_size_R"] * self.cfg["act_size"])
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.dict_size)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.dict_size, self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.dict_size,)).to(cfg["device"])

        self.to(device=cfg["device"], dtype=torch.float32)

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.intermediates = {}

    def forward(
        self,
        x,
        mask_features=None,
        compute_graph=True,
        mean_feature_acts=None,
        mean_error_acts=None,
        true_error_acts=None,
        fine_masking=False,
        mask_ratio=1.0
        ):
        if mask_features is not None: assert (mean_feature_acts is not None) and (mean_error_acts is not None)
        x = x[0, :, :]  # [1, B*T, D] -> [B*T, D]
        
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        ### >>> MODIFIED: to save z_before_acts
        z_before_acts = x_cent @ self.W_enc # [B*T, F]
        acts = F.relu(z_before_acts)  # [B*T, F]
        ### <<< MODIFIED: to save z_before_acts
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)  # (Indices, Values)
        if compute_graph:
            self.intermediates['acts_topk_raw'] = acts_topk
            
        acts_topk = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)  # [B*T, F]
        self.update_inactive_features(acts_topk)
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec  # [B*T, D]
        
        # For node ablation (insertion / deletion)
        x_reconstruct_with_mask = None
        if mask_features is not None and not self.training:

            if isinstance(mask_features, torch.Tensor):  # Ensure mask_features is long type and convert to tensor if necessary
                mask_features = mask_features.long()  # Ensure it's long type
            else:  # Convert list to tensor (if it's a list)
                mask_features = torch.tensor(mask_features, dtype=torch.long, device=self.cfg["device"])

            if not fine_masking:
                x_reconstruct_with_mask, apply_error_mask = self.get_recon_with_mask(acts_topk, mask_features, mean_feature_acts, mask_ratio)
            else:
                x_reconstruct_with_mask, apply_error_mask = self.get_recon_with_fine_mask(acts_topk, mask_features, mean_feature_acts)

            output = self.get_loss_dict(x, x_reconstruct_with_mask, acts, acts_topk, x_mean, x_std)
            output["sae_out_with_mask"] = self.postprocess_output(x_reconstruct_with_mask, x_mean, x_std)

            if apply_error_mask:  # Now true_error_acts is required
                output["sae_error"] = (1-mask_ratio) * true_error_acts + mask_ratio * mean_error_acts  # [B*T, D]
            else:
                if true_error_acts is not None:
                    output["sae_error"] = true_error_acts
                else:
                    output["sae_error"] = self.postprocess_output(x, x_mean, x_std) - output["sae_out"]  # x - \hat{x}  [B*T, D]
        
        else:
            output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
            output["sae_error"] = self.postprocess_output(x, x_mean, x_std) - output["sae_out"]
            output["sae_out_with_mask"] = None
        
        if compute_graph:
            self.intermediates['x_after_n'] = x_cent                # [B*T, D]
            self.intermediates['z_before_acts'] = z_before_acts     # [B*T, F]
            self.intermediates['z'] = acts_topk                     # [B*T, F]
            self.intermediates['x_hat_before_bn'] = x_reconstruct   # [B*T, D]
            self.intermediates['x_hat'] = output['sae_out']         # [B*T, D]
            self.intermediates['x_std'] = x_std                     # [B*T, 1]
            self.intermediates["sae_error"] = output["sae_error"]   # [B*T, D]
            
        return output

    
    def get_recon_with_mask(self, acts_topk, mask_features, mean_feature_acts, mask_ratio):

        num_nodes: int = acts_topk.shape[1]  # [F]
        apply_error_mask = False
        if num_nodes in mask_features:  # Checking if the error node index is in the mask
            mask_features = mask_features[mask_features != num_nodes]  # Remove the error node index
            apply_error_mask = True
        
        # Create the masked tensor
        acts_topk_masked = acts_topk.clone()
        acts_topk_masked[:, mask_features] = (1-mask_ratio) * acts_topk_masked[:, mask_features] + mask_ratio * mean_feature_acts[:, mask_features]
        x_reconstruct_with_mask = acts_topk_masked @ self.W_dec + self.b_dec  # [B*T, D]
        
        return x_reconstruct_with_mask, apply_error_mask


    def get_recon_with_fine_mask(self, acts_topk, mask_features, mean_feature_acts):

        num_nodes: int = acts_topk.shape[0] * acts_topk.shape[1]  # [T * F]
        apply_error_mask = False
        if num_nodes in mask_features:  # Checking if the error node index is in the mask
            mask_features = mask_features[mask_features != num_nodes]  # Remove the error node index
            apply_error_mask = True

        # Create the masked tensor
        acts_topk_masked = acts_topk.clone()
        rows, cols = mask_features // acts_topk.shape[1], mask_features % acts_topk.shape[1]
        acts_topk_masked[rows, cols] = mean_feature_acts[rows, cols]
        x_reconstruct_with_mask = acts_topk_masked @ self.W_dec + self.b_dec  # [B*T, D]

        return x_reconstruct_with_mask, apply_error_mask


    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()  # per element (div. by B*T * D)
        l1_norm = acts_topk.float().abs().sum(-1).mean()  # per instance (div. by B*T)
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()  # per instance (div. by B*T)
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std) # \hat{x}

        output = {
            "sae_out": sae_out,  # [B*T, D]
            "feature_acts": acts_topk,  # [B*T, F]
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)
