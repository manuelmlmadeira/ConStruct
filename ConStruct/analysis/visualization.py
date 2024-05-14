import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from ConStruct.metrics.sampling_molecular_metrics import Molecule
from ConStruct.utils import PlaceHolder
from ConStruct.datasets.tls_dataset import CellGraph


class Visualizer:
    def __init__(self, dataset_infos):
        self.dataset_infos = dataset_infos
        self.is_molecular = self.dataset_infos.is_molecular

        if self.is_molecular:
            self.remove_h = dataset_infos.remove_h

        self.is_tls = self.dataset_infos.is_tls

    def to_networkx(self, graph: PlaceHolder):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        nx_graph = nx.Graph()

        for i in range(len(graph.X)):
            nx_graph.add_node(i, number=i, symbol=graph.X[i], color_val=graph.X[i])

        adj = graph.E.cpu().numpy()
        assert len(adj.shape) == 2

        rows, cols = np.where(adj >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adj[edge[0], edge[1]]
            nx_graph.add_edge(
                edge[0], edge[1], color=float(edge_type), weight=3 * edge_type
            )

        return nx_graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100):
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        plt.figure()
        nx.draw(
            graph,
            pos,
            font_size=5,
            node_size=node_size,
            with_labels=False,
            node_color=U[:, 1],
            cmap=plt.cm.coolwarm,
            vmin=vmin,
            vmax=vmax,
            edge_color="grey",
        )

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(
        self,
        path: str,
        graphs: PlaceHolder,
        atom_decoder,
        num_graphs_to_visualize: int,
        log="graph",
    ):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        num_graphs = graphs.X.shape[0]
        num_graphs_to_visualize = min(num_graphs_to_visualize, num_graphs)
        if num_graphs_to_visualize > 0:
            print(f"Visualizing {num_graphs_to_visualize} graphs out of {num_graphs}")

        graph_list = graphs.split()
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, "graph_{}.png".format(i))

            if self.is_molecular:
                mol = Molecule(graph_list[i], atom_decoder).rdkit_mol
                try:
                    Draw.MolToFile(mol, file_path)
                except rdkit.Chem.KekulizeException:
                    print("Can't kekulize molecule")
            elif self.is_tls:
                cg = CellGraph.from_placeholder(graph_list[i])
                cg.plot_graph(save_path=file_path, has_legend=False)
            else:
                nx_graph = self.to_networkx(graph_list[i])
                self.visualize_non_molecule(graph=nx_graph, pos=None, path=file_path)

            if wandb.run and log is not None:
                if i < 3:
                    print(f"Saving {file_path} to wandb")
                wandb.log({log: [wandb.Image(file_path)]}, commit=False)

    def visualize_chains(
        self,
        chains: PlaceHolder,
        num_nodes: Tensor,
        chain_path: str,
        batch_id: int,
        local_rank: int,
    ):
        for i in range(chains.X.size(1)):  # Iterate over the chains
            path = os.path.join(chain_path, f"molecule_{batch_id + i}_{local_rank}")
            if not os.path.exists(path):
                os.makedirs(path)

            graphs = []
            chain = PlaceHolder(
                X=chains.X[:, i, : num_nodes[i]].long(),
                E=chains.E[:, i, : num_nodes[i], : num_nodes[i]].long(),
                charges=chains.charges[:, i, : num_nodes[i]].long(),
                y=None,
            )

            # Iterate over the frames of each molecule
            for j in range(chain.X.shape[0]):
                graph = PlaceHolder(
                    X=chain.X[j], E=chain.E[j], charges=chain.charges[j], y=None
                )
                if self.is_molecular:
                    graphs.append(
                        Molecule(
                            graph=graph, atom_decoder=self.dataset_infos.atom_decoder
                        )
                    )
                elif self.is_tls:
                    graphs.append(CellGraph.from_placeholder(graph))
                else:
                    graphs.append(self.to_networkx(graph))

            # Find the coordinates of nodes in the final graph and align all the molecules
            final_graph = graphs[-1]

            if self.is_molecular:
                final_mol = final_graph.rdkit_mol
                AllChem.Compute2DCoords(final_mol)
                coords = []
                for k, atom in enumerate(final_mol.GetAtoms()):
                    positions = final_mol.GetConformer().GetAtomPosition(k)
                    coords.append((positions.x, positions.y, positions.z))

                for graph in graphs:
                    mol = graph.rdkit_mol
                    AllChem.Compute2DCoords(mol)
                    conf = mol.GetConformer()
                    for l, atom in enumerate(mol.GetAtoms()):
                        x, y, z = coords[l]
                        conf.SetAtomPosition(l, Point3D(x, y, z))
            elif self.is_tls:
                final_graph.set_pos()
                final_pos = final_graph.get_pos()
            else:
                final_pos = nx.spring_layout(final_graph, seed=0)

            # Visualize and save
            save_paths = []
            for frame in range(len(graphs)):
                file_name = os.path.join(path, "frame_{}.png".format(frame))
                if self.is_molecular:
                    Draw.MolToFile(
                        graphs[frame].rdkit_mol,
                        file_name,
                        size=(300, 300),
                        legend=f"Frame {frame}",
                    )
                if self.is_tls:
                    if not graphs[frame].get_pos():  # The last one already has a pos
                        graphs[frame].set_pos(pos=final_pos)
                    graphs[frame].plot_graph(
                        save_path=file_name, has_legend=False, verbose=False
                    )
                else:
                    self.visualize_non_molecule(
                        graph=graphs[frame], pos=final_pos, path=file_name
                    )
                save_paths.append(file_name)

            print(
                f"{i + 1}/{chains.X.shape[1]} chains saved on local rank {local_rank}.",
                end="",
                flush=True,
            )

            imgs = [imageio.v3.imread(fn) for fn in save_paths]
            gif_path = os.path.join(
                os.path.dirname(path), "{}.gif".format(path.split("/")[-1])
            )
            imgs.extend([imgs[-1]] * 10)
            imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
            if wandb.run:
                wandb.log(
                    {"chain": [wandb.Video(gif_path, caption=gif_path, format="gif")]}
                )
                print(f"Saving {gif_path} to wandb")
                wandb.log(
                    {"chain": wandb.Video(gif_path, fps=8, format="gif")}, commit=True
                )
