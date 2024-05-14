import os

import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops
from torchmetrics import (
    Metric,
    MeanSquaredError,
    MeanAbsoluteError,
    MetricCollection,
    KLDivergence,
)
from omegaconf import OmegaConf
import wandb


class NoSyncMetricCollection(MetricCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )  # disabling syncs since it messes up DDP sub-batching


class NoSyncMetric(Metric):
    def __init__(self):
        super().__init__(
            sync_on_compute=False, dist_sync_on_step=False
        )  # disabling syncs since it messes up DDP sub-batching


class NoSyncKL(KLDivergence):
    def __init__(self):
        super().__init__(
            sync_on_compute=False, dist_sync_on_step=False
        )  # disabling syncs since it messes up DDP sub-batching


class NoSyncMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(
            sync_on_compute=False, dist_sync_on_step=False
        )  # disabling syncs since it messes up DDP sub-batching


class NoSyncMAE(MeanAbsoluteError):
    def __init__(self):
        super().__init__(
            sync_on_compute=False, dist_sync_on_step=False
        )  # disabling syncs since it messes up DDP sub-batching>>>>>>> main:utils.py


# Folders
def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs("graphs", exist_ok=True)
        os.makedirs("chains", exist_ok=True)
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs("graphs/" + args.general.name, exist_ok=True)
        os.makedirs("chains/" + args.general.name, exist_ok=True)
    except OSError:
        pass


def to_dense(data, dataset_info, device=None):
    X, node_mask = to_dense_batch(x=data.x, batch=data.batch)
    charges, _ = (
        to_dense_batch(x=data.charges, batch=data.batch)
        if hasattr(data, "charges")  # Spectre datasets do not have charges
        else (None, None)
    )

    max_num_nodes = X.size(1)
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=data.batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )

    dense_data = PlaceHolder(X=X, charges=charges, E=E, node_mask=node_mask, y=data.y)

    dense_data = dataset_info.to_one_hot(dense_data)
    dense_data.y = dense_data.X.new_zeros((X.shape[0], 0))
    dense_data = dense_data.device_as(X.to(device))

    return dense_data.mask()


class PlaceHolder:
    def __init__(self, X, E, y, charges=None, t_int=None, t=None, node_mask=None):
        self.X = X
        self.charges = charges
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask

    def device_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.to(x.device) if self.X is not None else None
        self.charges = self.charges.to(x.device) if self.charges is not None else None
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None
        return self

    def mask(self, node_mask=None):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        diag_mask = (
            ~torch.eye(n, dtype=torch.bool, device=node_mask.device)
            .unsqueeze(0)
            .expand(bs, -1, -1)
            .unsqueeze(-1)
        )  # bs, n, n, 1

        if self.X is not None:
            self.X = self.X * x_mask
        if self.charges is not None and self.charges.numel() > 0:
            self.charges = self.charges * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
        assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def collapse(self, collapse_charges):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        if self.charges.numel() > 0:
            copy.charges = collapse_charges.to(self.charges.device)[
                torch.argmax(self.charges, dim=-1)
            ]
        else:
            copy.charges = self.charges.new_zeros((self.charges.shape[:-1]))
        copy.E = torch.argmax(self.E, dim=-1)
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = -1
        copy.charges[self.node_mask == 0] = 1000
        copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1

        # TODO: this diag mask might be important to get the dataset statistics correct
        # diag_mask = (
        #     ~torch.eye(e_mask1.shape[1], device=x_mask.device, dtype=torch.bool)
        #     .unsqueeze(0)
        #     .repeat(e_mask1.shape[0], 1, 1)
        # )
        # copy.E[~diag_mask] = -1
        return copy

    def split(self):
        """Split a PlaceHolder representing a batch into a list of placeholders representing individual graphs."""
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(self.node_mask[i], dim=0)
            x = self.X[i, :n]
            c = self.charges[i, :n]
            e = self.E[i, :n, :n]
            graph_list.append(
                PlaceHolder(X=x, charges=c, E=e, y=self.y[i], node_mask=None)
            )
        return graph_list

    def __repr__(self):
        return (
            f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"charges: {self.charges.shape if type(self.charges) == torch.Tensor else self.charges} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def copy(self):
        return PlaceHolder(
            X=self.X,
            charges=self.charges,
            E=self.E,
            y=self.y,
            t_int=self.t_int,
            t=self.t,
            node_mask=self.node_mask,
        )


def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_dict["general"]["local_dir"] = os.getcwd()
    dataset_name = cfg.dataset["name"]
    if dataset_name == "qm9" and not cfg.dataset.remove_h:
        dataset_name = "qm9_h"
    kwargs = {
        "name": cfg.general.name,
        "project": f"DiGressV2_{dataset_name}",
        "config": config_dict,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.general.wandb,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")
    return cfg
