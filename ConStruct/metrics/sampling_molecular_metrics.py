import os
from collections import Counter

from rdkit import Chem, RDLogger
from torchmetrics import MeanMetric, MeanAbsoluteError, Metric, CatMetric
import torch
import wandb
import torch.nn as nn

from ConStruct.utils import PlaceHolder
import fcd
import numpy as np

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

RDLogger.DisableLog("rdApp.*")


class Molecule:
    def __init__(self, graph: PlaceHolder, atom_decoder):
        """atom_decoder: extracted from dataset_infos."""
        self.atom_types = graph.X.long()
        self.bond_types = graph.E.long()
        self.charges = graph.charges.long()
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.atom_decoder = atom_decoder
        self.num_nodes = len(graph.X)
        self.num_atom_types = len(atom_decoder)
        self.device = self.atom_types.device

    def build_molecule(self, atom_decoder):
        mol = Chem.RWMol()
        for atom, charge in zip(self.atom_types, self.charges):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            a.SetFormalCharge(charge.item())
            mol.AddAtom(a)
        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None
        return mol

    def check_stability(self, debug=False):
        e = self.bond_types.clone()
        e[e == 4] = 1.5
        e[e < 0] = 0
        valencies = torch.sum(e, dim=-1).long()

        n_stable_at = 0
        mol_stable = True
        for i, (atom_type, valency, charge) in enumerate(
            zip(self.atom_types, valencies, self.charges)
        ):
            atom_type = atom_type.item()
            valency = valency.item()
            charge = charge.item()
            possible_bonds = allowed_bonds[self.atom_decoder[atom_type]]
            if type(possible_bonds) == int:
                is_stable = possible_bonds == valency
            elif type(possible_bonds) == dict:
                expected_bonds = (
                    possible_bonds[charge]
                    if charge in possible_bonds.keys()
                    else possible_bonds[0]
                )
                is_stable = (
                    expected_bonds == valency
                    if type(expected_bonds) == int
                    else valency in expected_bonds
                )
            else:
                is_stable = valency in possible_bonds
            if not is_stable:
                mol_stable = False
            if not is_stable and debug:
                print(
                    f"Invalid atom {self.atom_decoder[atom_type]}: valency={valency}, charge={charge}"
                )
                print()
            n_stable_at += int(is_stable)

        return mol_stable, n_stable_at, len(self.atom_types)


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, test):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.is_molecular = dataset_infos.is_molecular
        self.atom_decoder = dataset_infos.atom_decoder

        self.test = test

        self.atom_stable = MeanMetric()
        self.mol_stable = MeanMetric()

        # Retrieve dataset smiles only for qm9 currently.
        self.train_smiles = set(dataset_infos.train_smiles)
        self.validity_metric = MeanMetric()
        # self.uniqueness_metric = UniquenessMetric()
        self.novelty_metric = MeanMetric(nan_strategy="ignore")

        self.charge_w1 = MeanMetric()
        self.valency_w1 = MeanMetric()

        # for fcd
        self.val_smiles = list(
            dataset_infos.test_smiles if test else dataset_infos.val_smiles
        )
        self.val_fcd_mu, self.val_fcd_sigma = (
            dataset_infos.test_fcd_stats if test else dataset_infos.val_fcd_stats
        )

    def reset(self):
        for metric in [
            self.atom_stable,
            self.mol_stable,
            self.validity_metric,  # self.uniqueness_metric,
            self.novelty_metric,
            self.atom_stable,
            self.mol_stable,
            self.charge_w1,
            self.valency_w1,
        ]:
            metric.reset()

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        all_smiles = []
        error_message = Counter()
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        rdmol, asMols=True, sanitizeFrags=True
                    )
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol, canonical=True)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    all_smiles.append("error")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    all_smiles.append("error")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
                    all_smiles.append("error")
        print(
            f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
            f" -- No error {error_message[-1]}"
        )
        self.validity_metric.update(
            value=len(valid) / len(generated), weight=len(generated)
        )
        return valid, all_smiles, error_message

    def evaluate(self, generated):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""

        valid_mols, all_smiles, error_message = self.compute_validity(generated)
        validity = self.validity_metric.compute().item()

        # self.uniqueness_metric.update(valid_mols)
        if len(valid_mols) > 0:
            uniqueness = len(set(valid_mols)) / len(valid_mols)
        else:
            uniqueness = 0.0
        unique = list(set(valid_mols))

        if self.train_smiles is not None and len(unique) > 0:
            novel = [s for s in unique if s not in self.train_smiles]
            self.novelty_metric.update(
                value=len(novel) / len(unique), weight=len(unique)
            )
            novelty = self.novelty_metric.compute().item()
        else:
            print("No valid molecules")
            novelty = 0.0
        # num_molecules = int(self.validity_metric.weight.item())
        # print(f"Validity over {num_molecules} molecules: {validity * 100 :.2f}%")

        key = "val_sampling" if not self.test else "test_sampling"
        dic = {
            f"{key}/Validity": validity * 100,
            f"{key}/Uniqueness": uniqueness * 100 if uniqueness != 0 else 0,
            f"{key}/Novelty": novelty * 100 if novelty != 0 else 0,
        }

        return all_smiles, dic

    def forward(self, generated_graphs: list[PlaceHolder], current_epoch, local_rank):
        molecules = []

        for batch in generated_graphs:
            graphs = batch.split()
            for graph in graphs:
                molecule = Molecule(graph, atom_decoder=self.dataset_infos.atom_decoder)
                molecules.append(molecule)

        if not self.dataset_infos.remove_h:
            print(f"Analyzing molecule stability on {local_rank}...")
            for i, mol in enumerate(molecules):
                mol_stable, at_stable, num_bonds = mol.check_stability()
                self.mol_stable.update(value=mol_stable)
                self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

            stability_dict = {
                "mol_stable": self.mol_stable.compute().item(),
                "atm_stable": self.atom_stable.compute().item(),
            }
            if local_rank == 0:
                print("Stability metrics:", stability_dict)
                if wandb.run:
                    wandb.log(stability_dict, commit=False)

        # Validity, uniqueness, novelty
        all_generated_smiles, metrics = self.evaluate(molecules)
        if len(all_generated_smiles) > 0 and local_rank == 0:
            print("Some generated smiles: " + " ".join(all_generated_smiles[:10]))

        to_log_fcd = self.compute_fcd(generated_smiles=all_generated_smiles)
        metrics.update(to_log_fcd)

        # Save in any case in the graphs folder
        os.makedirs("graphs", exist_ok=True)
        textfile = open(
            f"graphs/valid_unique_molecules_e{current_epoch}_GR{local_rank}.txt", "w"
        )
        textfile.writelines(all_generated_smiles)
        textfile.close()
        # Save in the root folder if test_model
        if self.test:
            filename = f"final_smiles_GR{local_rank}_{0}.txt"
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f"final_smiles_GR{local_rank}_{i}.txt"
                else:
                    break
            with open(filename, "w") as fp:
                for smiles in all_generated_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print(f"All smiles saved on rank {local_rank}")
        # Compute statistics
        stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )

        key = "val_sampling" if not self.test else "test_sampling"
        if self.is_molecular:
            # the molecule validity is calculated with charge information
            charge_w1, charge_w1_per_class = charge_distance(
                molecules, stat.charge_types, stat.atom_types, self.dataset_infos
            )
            self.charge_w1(charge_w1)
            metrics[f"{key}/ChargeW1"] = self.charge_w1.compute().item()
            valency_w1, valency_w1_per_class = valency_distance(
                molecules,
                stat.valencies,
                stat.atom_types,
                self.dataset_infos.atom_encoder,
            )
            self.valency_w1(valency_w1)
            # TODO: (line below) torch lightning stalls for multi-gpu sampling is number of samples to generate is <=2
            metrics[f"{key}/ValencyW1"] = self.valency_w1.compute().item()

        # if local_rank == 0:
        #     print(f"Sampling metrics", {k: round(val, 3) for k, val in metrics.items()})
        if local_rank == 0:
            print(f"Molecular metrics computed.")
        return metrics

    def compute_fcd(self, generated_smiles):
        fcd_model = fcd.load_ref_model()
        generated_smiles = [
            smile
            for smile in fcd.canonical_smiles(generated_smiles)
            if smile is not None
        ]

        if len(generated_smiles) <= 1:
            print("Not enough (<=1) valid smiles for FCD computation.")
            fcd_score = -1
        else:
            gen_activations = fcd.get_predictions(fcd_model, generated_smiles)
            gen_mu = np.mean(gen_activations, axis=0)
            gen_sigma = np.cov(gen_activations.T)
            target_mu = self.val_fcd_mu
            target_sigma = self.val_fcd_sigma
            try:
                fcd_score = fcd.calculate_frechet_distance(
                    mu1=gen_mu,
                    sigma1=gen_sigma,
                    mu2=target_mu,
                    sigma2=target_sigma,
                )
            except ValueError as e:
                eps = 1e-6
                print(f"Error in FCD computation: {e}. Increasing eps to {eps}")
                eps_sigma = eps * np.eye(gen_sigma.shape[0])
                gen_sigma = gen_sigma + eps_sigma
                target_sigma = self.val_fcd_sigma + eps_sigma
                fcd_score = fcd.calculate_frechet_distance(
                    mu1=gen_mu,
                    sigma1=gen_sigma,
                    mu2=target_mu,
                    sigma2=target_sigma,
                )

        print(f"FCD score: {fcd_score}")
        key = "val_sampling" if not self.test else "test_sampling"
        return {f"{key}/fcd score": fcd_score}


def charge_distance(molecules, target, atom_types_probabilities, dataset_infos):
    device = molecules[0].bond_types.device
    generated_distribution = torch.zeros_like(target).to(device)
    for molecule in molecules:
        for atom_type in range(target.shape[0]):
            mask = molecule.atom_types == atom_type
            if mask.sum() > 0:
                at_charges = dataset_infos.one_hot_charges(molecule.charges[mask])
                generated_distribution[atom_type] += at_charges.sum(dim=0)

    s = generated_distribution.sum(dim=1, keepdim=True)
    s[s == 0] = 1
    generated_distribution = generated_distribution / s

    cs_generated = torch.cumsum(generated_distribution, dim=1)
    cs_target = torch.cumsum(target, dim=1).to(device)

    w1_per_class = torch.sum(torch.abs(cs_generated - cs_target), dim=1)

    w1 = torch.sum(w1_per_class * atom_types_probabilities.to(device)).item()
    return w1, w1_per_class


def valency_distance(
    molecules, target_valencies, atom_types_probabilities, atom_encoder
):
    # Build a dict for the generated molecules that is similar to the target one
    num_atom_types = len(atom_types_probabilities)
    generated_valencies = {i: Counter() for i in range(num_atom_types)}
    for molecule in molecules:
        edge_types = molecule.bond_types
        edge_types[edge_types == 4] = 1.5
        valencies = torch.sum(edge_types, dim=0)
        for atom, val in zip(molecule.atom_types, valencies):
            generated_valencies[atom.item()][val.item()] += 1

    # Convert the valencies to a tensor of shape (num_atom_types, max_valency)
    max_valency_target = max(
        max(vals.keys()) if len(vals) > 0 else -1 for vals in target_valencies.values()
    )
    max_valency_generated = max(
        max(vals.keys()) if len(vals) > 0 else -1
        for vals in generated_valencies.values()
    )
    max_valency = int(max(max_valency_target, max_valency_generated))

    valencies_target_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in target_valencies.items():
        for valency, count in valencies.items():
            valencies_target_tensor[int(atom_encoder[atom_type]), int(valency)] = count

    valencies_generated_tensor = torch.zeros(num_atom_types, max_valency + 1)
    for atom_type, valencies in generated_valencies.items():
        for valency, count in valencies.items():
            valencies_generated_tensor[int(atom_type), int(valency)] = count

    # Normalize the distributions
    s1 = torch.sum(valencies_target_tensor, dim=1, keepdim=True)
    s1[s1 == 0] = 1
    valencies_target_tensor = valencies_target_tensor / s1

    s2 = torch.sum(valencies_generated_tensor, dim=1, keepdim=True)
    s2[s2 == 0] = 1
    valencies_generated_tensor = valencies_generated_tensor / s2

    cs_target = torch.cumsum(valencies_target_tensor, dim=1)
    cs_generated = torch.cumsum(valencies_generated_tensor, dim=1)

    w1_per_class = torch.sum(torch.abs(cs_target - cs_generated), dim=1)

    # print('debugging for molecular_metrics - valency_distance')
    # print('cs_target', cs_target)
    # print('cs_generated', cs_generated)

    total_w1 = torch.sum(w1_per_class * atom_types_probabilities).item()
    return total_w1, w1_per_class


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class MeanNumberEdge(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)


#
# class UniquenessMetric(Metric):
#     is_differentiable = False
#     higher_is_better = True
#     full_state_update = True
#     def __init__(self):
#         """ Check if the number of unique molecules by concatenating the smiles. """
#         super().__init__(compute_on_cpu=True)
#         # add the smiles as a state
#         self.add_state("smiles", default=[], dist_reduce_fx=None)
#
#     def update(self, smiles_list):
#         self.smiles.extend(smiles_list)
#
#     def compute(self):
#         print(f"Computing uniqueness over {len(self.smiles)} smiles")
#         if len(self.smiles) == 0:
#             return 0.0
#         return len(set(self.smiles)) / len(self.smiles)


def smiles_from_generated_samples_file(generated_samples_file, atom_decoder):
    """ """
    smiles_list = []
    with open(generated_samples_file, "r") as f:
        # Repeat until the end of the file
        while True:
            # Check if we reached the end of the file
            line = f.readline()
            print("First line", line)
            if not line:
                break
            else:
                N = int(line.split("=")[1])

            # Extract X (labels or coordinates of the nodes)
            f.readline()
            X = list(map(int, f.readline().split()))
            X = torch.tensor(X)[:N]

            # Extract charges
            f.readline()
            charges = list(map(int, f.readline().split()))
            charges = torch.tensor(charges)
            N_before_mask = len(charges)
            charges = charges[:N]

            f.readline()
            E = []
            for i in range(N_before_mask):
                E.append(list(map(int, f.readline().split())))
            E = torch.tensor(E)[:N, :N]

            graph = PlaceHolder(X=X, E=E, charges=charges, y=None)
            f.readline()
            mol = Molecule(graph, atom_decoder)
            rdkit_mol = mol.rdkit_mol
            # Try to convert to smiles
            try:
                smiles = Chem.MolToSmiles(rdkit_mol)
            except:
                smiles = None
            print(smiles)
            smiles_list.append(smiles)

        # Save the smiles list to file
        with open(generated_samples_file.split(".")[0] + ".smiles", "w") as f:
            for smiles in smiles_list:
                f.write(smiles + "\n")
    return smiles


if __name__ == "__main__":
    file = ""
    atom_decoder = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

    smiles_list = smiles_from_generated_samples_file(file, atom_decoder)
    print(smiles_list)
