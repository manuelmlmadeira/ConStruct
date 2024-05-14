import os
import os.path as osp
import pathlib


import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd

from ConStruct.utils import PlaceHolder
from ConStruct.datasets.abstract_dataset import (
    MolecularDataModule,
    AbstractDatasetInfos,
)
from ConStruct.datasets.dataset_utils import (
    save_pickle,
    mol_to_torch_geometric,
    load_pickle,
    Statistics,
    compute_reference_metrics,
)
from ConStruct.metrics.metrics_utils import compute_all_statistics
import fcd


atom_encoder = {"C": 0, "N": 1, "S": 2, "O": 3, "F": 4, "Cl": 5, "Br": 6}
atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br"]


class MosesDataset(InMemoryDataset):
    train_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv"
    val_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv"
    test_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv"

    def __init__(
        self,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        self.atom_encoder = atom_encoder
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            degree_hist=load_pickle(self.processed_paths[4]),
            charge_types=torch.from_numpy(np.load(self.processed_paths[5])).float(),
            valencies=load_pickle(self.processed_paths[6]),
        )
        self.smiles = load_pickle(self.processed_paths[7])
        if self.split in ["val", "test"]:
            fcd_stats = np.load(self.processed_paths[8])
            self.mu_fcd, self.sigma_fcd = fcd_stats["mu_fcd"], fcd_stats["sigma_fcd"]

    @property
    def raw_file_names(self):
        return ["train_moses.csv", "val_moses.csv", "test_moses.csv"]

    @property
    def split_file_name(self):
        return ["train_moses.csv", "val_moses.csv", "test_moses.csv"]

    @property
    def processed_file_names(self):
        f = ""  # Legacy
        if self.split == "train":
            return [
                f"train_{f}.pt",
                f"train_n_{f}.pickle",
                f"train_atom_types_{f}.npy",
                f"train_bond_types_{f}.npy",
                f"train_degrees.pickle",
                f"train_charges_{f}.npy",
                f"train_valency_{f}.pickle",
                f"train_smiles_{f}.pickle",
            ]
        elif self.split == "val":
            return [
                f"val_{f}.pt",
                f"val_n_{f}.pickle",
                f"val_atom_types_{f}.npy",
                f"val_bond_types_{f}.npy",
                f"val_degrees.pickle",
                f"val_charges_{f}.npy",
                f"val_valency_{f}.pickle",
                "val_smiles.pickle",
                f"val_fcd_stats.npz",
            ]
        else:
            return [
                f"test_{f}.pt",
                f"test_n_{f}.pickle",
                f"test_atom_types_{f}.npy",
                f"test_bond_types_{f}.npy",
                f"test_degrees.pickle",
                f"test_charges_{f}.npy",
                f"test_valency_{f}.pickle",
                "test_smiles.pickle",
                f"test_fcd_stats.npz",
            ]

    def download(self):
        import rdkit  # noqa

        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, "train_moses.csv"))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, "val_moses.csv"))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, "test_moses.csv"))

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        smile_list = pd.read_csv(self.raw_paths[self.file_idx])
        smile_list = smile_list["SMILES"].values
        data_list = []
        smiles_kept = []
        charges_list = set()

        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)

            if mol is not None:
                data = mol_to_torch_geometric(mol, atom_encoder, smile)
                unique_charges = set(torch.unique(data.charges).int().numpy())
                charges_list = charges_list.union(unique_charges)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                smiles_kept.append(smile)

        statistics = compute_all_statistics(
            data_list, self.atom_encoder, charges_dic={0: 0}
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        save_pickle(statistics.degree_hist, self.processed_paths[4])
        np.save(self.processed_paths[5], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[6])
        print(
            "Number of molecules that could not be mapped to smiles: ",
            len(smile_list) - len(smiles_kept),
        )
        save_pickle(set(smiles_kept), self.processed_paths[7])
        torch.save(self.collate(data_list), self.processed_paths[0])

        if self.split in ["val", "test"]:
            # The ones being compared in FCD
            # Delete repeated and invalid molecules (smile is None)
            print("Canonicalizing smiles for FCD")
            smiles_kept = list(
                set(
                    [
                        smile
                        for smile in fcd.canonical_smiles(smiles_kept)
                        if smile is not None
                    ]
                )
            )
            print("Computing FCD activations")
            fcd_model = fcd.load_ref_model()
            activations = fcd.get_predictions(fcd_model, smiles_kept)
            print("Computing FCD statistics")
            mu_fcd = np.mean(activations, axis=0)
            sigma_fcd = np.cov(activations.T)
            # Save FCD statistics
            np.savez(self.processed_paths[8], mu_fcd=mu_fcd, sigma_fcd=sigma_fcd)


class MosesDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        self.remove_h = False

        datasets = {
            key: MosesDataset(split=key, root=root_path)
            for key in ["train", "val", "test"]
        }
        self.statistics = {
            key: datasets[key].statistics for key in ["train", "val", "test"]
        }

        super().__init__(cfg, datasets)


class MosesInfos(AbstractDatasetInfos):
    """
    Moses will not support charge as it only contains one charge type 1
    """

    def __init__(self, datamodule, cfg):
        # basic information
        self.name = "moses"
        self.is_molecular = True
        self.is_tls = False
        self.remove_h = False
        # statistics
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.statistics = datamodule.statistics
        self.collapse_charges = torch.Tensor([0]).int()
        self.train_smiles = datamodule.train_dataset.smiles
        super().complete_infos(datamodule.statistics)

        # data specific settings
        # atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
        self.valencies = [4, 3, 2, 2, 1, 1, 1]
        self.atom_weights = [12, 14, 32, 16, 19, 35.4, 79.9]
        self.max_weight = 9 * 80  # Quite arbitrary

        # FCD stuff
        self.val_smiles = datamodule.val_dataset.smiles
        self.test_smiles = datamodule.test_dataset.smiles
        self.val_fcd_stats = (
            datamodule.val_dataset.mu_fcd,
            datamodule.val_dataset.sigma_fcd,
        )
        self.test_fcd_stats = (
            datamodule.test_dataset.mu_fcd,
            datamodule.test_dataset.sigma_fcd,
        )
        compute_reference_metrics(self, datamodule)
