import os
import os.path as osp
import pathlib

import hashlib
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


TRAIN_HASH = "05ad85d871958a05c02ab51a4fde8530"
VALID_HASH = "e53db4bff7dc4784123ae6df72e3b1f0"
TEST_HASH = "677b757ccec4809febd83850b43e1616"


atom_encoder = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "B": 4,
    "Br": 5,
    "Cl": 6,
    "I": 7,
    "P": 8,
    "S": 9,
    "Se": 10,
    "Si": 11,
}
atom_decoder = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]


def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, "rb").read()).hexdigest()
    if output_hash != correct_hash:
        print(
            f"{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!"
        )
        return False

    return True


class GuacamolDataset(InMemoryDataset):
    train_url = "https://figshare.com/ndownloader/files/13612760"
    test_url = "https://figshare.com/ndownloader/files/13612757"
    valid_url = "https://figshare.com/ndownloader/files/13612766"
    all_url = "https://figshare.com/ndownloader/files/13612745"

    def __init__(
        self,
        split,
        root,
        filter_dataset: bool,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        self.filter_dataset = filter_dataset
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
        return [
            "guacamol_v1_train.smiles",
            "guacamol_v1_valid.smiles",
            "guacamol_v1_test.smiles",
        ]

    @property
    def processed_file_names(self):
        f = "f" if self.filter_dataset else ""
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
                f"val_smiles_{f}.pickle",
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
                f"test_smiles_{f}.pickle",
                f"test_fcd_stats.npz",
            ]

    def download(self):
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, "guacamol_v1_train.smiles"))
        train_path = osp.join(self.raw_dir, "guacamol_v1_train.smiles")

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, "guacamol_v1_test.smiles"))
        test_path = osp.join(self.raw_dir, "guacamol_v1_test.smiles")

        valid_path = download_url(self.valid_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, "guacamol_v1_valid.smiles"))
        valid_path = osp.join(self.raw_dir, "guacamol_v1_valid.smiles")

        # check the hashes
        # Check whether the md5-hashes of the generated smiles files match
        # the precomputed hashes, this ensures everyone works with the same splits.
        valid_hashes = [
            compare_hash(train_path, TRAIN_HASH),
            compare_hash(valid_path, VALID_HASH),
            compare_hash(test_path, TEST_HASH),
        ]

        if not all(valid_hashes):
            raise SystemExit("Invalid hashes for the dataset files")

        print("Dataset download successful. Hashes are correct.")

        # if files_exist(self.split_paths):
        #     return

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        smile_list = open(self.raw_paths[self.file_idx]).readlines()
        data_list = []
        smiles_kept = []

        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)

            if mol is not None:
                data = mol_to_torch_geometric(mol, atom_encoder, smile)
                # data.y = torch.zeros(size=(1, 0), dtype=torch.float)
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                if self.filter_dataset and self.split == "train":
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(
                            mol, asMols=True, sanitizeFrags=True
                        )
                        if len(mol_frags) == 1:
                            smiles = Chem.MolToSmiles(mol)
                            if self.pre_filter is not None and not self.pre_filter(
                                data
                            ):
                                continue
                            if self.pre_transform is not None:
                                data = self.pre_transform(data)
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")

                else:
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
                    smiles_kept.append(smile)

            # if i > 1000:
            #     break

        statistics = compute_all_statistics(
            data_list, self.atom_encoder, charges_dic={-1: 0, 0: 1, 1: 2, 2: 3, 3: 4}
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        save_pickle(statistics.degree_hist, self.processed_paths[4])
        np.save(self.processed_paths[5], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[6])
        save_pickle(set(smiles_kept), self.processed_paths[7])
        torch.save(self.collate(data_list), self.processed_paths[0])
        print(
            "Number of molecules that could not be mapped to smiles: ",
            len(smile_list) - len(smiles_kept),
        )

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


class GuacamolDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.remove_h = True
        self.datadir = cfg.dataset.datadir
        self.filter = cfg.dataset.filter
        self.train_smiles = []
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            "train": GuacamolDataset(
                split="train", root=root_path, filter_dataset=self.filter
            ),
            "val": GuacamolDataset(
                split="val", root=root_path, filter_dataset=self.filter
            ),
            "test": GuacamolDataset(
                split="test", root=root_path, filter_dataset=self.filter
            ),
        }

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }
        super().__init__(cfg, datasets)


class GuacamolInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        # basic information and settings
        self.name = "guacamol"
        self.is_molecular = True
        self.is_tls = False
        self.remove_h = cfg.dataset.remove_h

        # other statistics
        self.statistics = datamodule.statistics
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.collapse_charges = torch.Tensor([-1, 0, 1, 2, 3]).int()
        self.train_smiles = datamodule.train_dataset.smiles
        super().complete_infos(datamodule.statistics)

        # dataset specific settings
        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]
        self.atom_weights = [
            12,
            14,
            16,
            19,
            10.81,
            79.9,
            35.45,
            126.9,
            30.97,
            32.06,
            78.97,
            28.09,
        ]
        self.max_weight = 1000

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
