import matplotlib.hatch
import torch
import torch.nn.functional as F
import numpy as np

import ConStruct.utils as utils
from ConStruct.diffusion import diffusion_utils


class NoiseModel:
    def __init__(self, cfg):
        self.mapping = ["x", "c", "e", "y"]
        self.inverse_mapping = {m: i for i, m in enumerate(self.mapping)}
        nu = cfg.model.nu
        self.nu_arr = []
        for m in self.mapping:
            self.nu_arr.append(nu[m])

        # Define the transition matrices for the discrete features
        self.Px = None
        self.Pe = None
        self.Py = None
        self.Pcharges = None
        self.X_classes = None
        self.charges_classes = None
        self.E_classes = None
        self.y_classes = None
        self.X_marginals = None
        self.charges_marginals = None
        self.E_marginals = None
        self.y_marginals = None

        self.timesteps = cfg.model.diffusion_steps
        self.T = cfg.model.diffusion_steps

        if cfg.model.transition in ["uniform", "marginal", "planar", "absorbing_edges"]:
            betas = diffusion_utils.cosine_beta_schedule_discrete(
                self.timesteps, self.nu_arr
            )
        elif cfg.model.transition in ["absorbing"]:
            betas = diffusion_utils.linear_beta_schedule(self.timesteps, self.nu_arr)
        else:
            raise NotImplementedError(self.model.transition)

        self._betas = torch.from_numpy(betas)
        self._alphas = 1 - torch.clamp(self._betas, min=0, max=0.9999)
        log_alpha = torch.log(self._alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self._log_alpha_bar = log_alpha_bar
        self._alphas_bar = torch.exp(log_alpha_bar)
        self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
        self._sigma_bar = torch.sqrt(self._sigma2_bar)
        self._gamma = (
            torch.log(-torch.special.expm1(2 * log_alpha_bar)) - 2 * log_alpha_bar
        )
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def complete_init(self):
        """Compute the transition matrices"""
        self.Px = self.X_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.Pe = self.E_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.Pcharges = (
            self.charges_marginals.unsqueeze(0)
            .expand(self.charges_classes, -1)
            .unsqueeze(0)
        )
        self.Py = torch.ones(1, self.y_classes, self.y_classes) / self.y_classes

    def move_P_device(self, tensor):
        """Move the transition matrices to the device specified by tensor."""
        return diffusion_utils.PlaceHolder(
            X=self.Px.float().to(tensor.device),
            charges=self.Pcharges.float().to(tensor.device),
            E=self.Pe.float().to(tensor.device).float(),
            y=self.Py.float().to(tensor.device),
        )

    def get_Qt(self, t_int):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        P = self.move_P_device(t_int)
        kwargs = {"device": t_int.device, "dtype": torch.float32}

        bx = self.get_beta(t_int=t_int, key="x").unsqueeze(1)
        q_x = bx * P.X + (1 - bx) * torch.eye(self.X_classes, **kwargs).unsqueeze(0)

        bc = self.get_beta(t_int=t_int, key="c").unsqueeze(1)
        q_c = bc * P.charges + (1 - bc) * torch.eye(
            self.charges_classes, **kwargs
        ).unsqueeze(0)

        be = self.get_beta(t_int=t_int, key="e").unsqueeze(1)
        q_e = be * P.E + (1 - be) * torch.eye(self.E_classes, **kwargs).unsqueeze(0)

        by = self.get_beta(t_int=t_int, key="y").unsqueeze(1)
        q_y = by * P.y + (1 - by) * torch.eye(self.y_classes, **kwargs).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, charges=q_c, E=q_e, y=q_y)

    def get_Qt_bar(self, t_int):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        a_x = self.get_alpha_bar(t_int=t_int, key="x").unsqueeze(1)
        a_c = self.get_alpha_bar(t_int=t_int, key="c").unsqueeze(1)
        a_e = self.get_alpha_bar(t_int=t_int, key="e").unsqueeze(1)
        a_y = self.get_alpha_bar(t_int=t_int, key="y").unsqueeze(1)

        P = self.move_P_device(t_int)
        # [X, charges, E, y, pos]
        dev = t_int.device
        q_x = a_x * torch.eye(self.X_classes, device=dev).unsqueeze(0) + (1 - a_x) * P.X
        q_c = (
            a_c * torch.eye(self.charges_classes, device=dev).unsqueeze(0)
            + (1 - a_c) * P.charges
        )
        q_e = a_e * torch.eye(self.E_classes, device=dev).unsqueeze(0) + (1 - a_e) * P.E
        q_y = a_y * torch.eye(self.y_classes, device=dev).unsqueeze(0) + (1 - a_y) * P.y

        assert ((q_x.sum(dim=2) - 1.0).abs() < 1e-4).all()
        assert ((q_e.sum(dim=2) - 1.0).abs() < 1e-4).all()
        assert ((q_c.sum(dim=2) - 1.0).abs() < 1e-4).all()

        return utils.PlaceHolder(X=q_x, charges=q_c, E=q_e, y=q_y)

    def _get(self, self_series, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        att = self_series.to(t_int.device)[t_int.long()]
        if key is None:
            return att.float()
        else:
            return att[..., self.inverse_mapping[key]].float()

    def get_beta(self, t_normalized=None, t_int=None, key=None):
        return self._get(self._betas, t_normalized=t_normalized, t_int=t_int, key=key)

    def get_alpha_bar(self, t_normalized=None, t_int=None, key=None):
        return self._get(
            self._alphas_bar, t_normalized=t_normalized, t_int=t_int, key=key
        )

    def get_sigma_bar(self, t_normalized=None, t_int=None, key=None):
        return self._get(
            self._sigma_bar, t_normalized=t_normalized, t_int=t_int, key=key
        )

    def get_sigma2_bar(self, t_normalized=None, t_int=None, key=None):
        return self._get(
            self._sigma2_bar, t_normalized=t_normalized, t_int=t_int, key=key
        )

    def get_gamma(self, t_normalized=None, t_int=None, key=None):
        return self._get(self._gamma, t_normalized=t_normalized, t_int=t_int, key=key)

    def apply_noise(self, dense_data):
        """Sample noise and apply it to the data."""
        device = dense_data.X.device
        t_int = torch.randint(
            1, self.T + 1, size=(dense_data.X.size(0), 1), device=device
        )
        t_float = t_int.float() / self.T

        # Qtb returns two matrices of shape (bs, dx_in, dx_out) and (bs, de_in, de_out)
        Qtb = self.get_Qt_bar(t_int=t_int)

        # Compute transition probabilities
        probX = dense_data.X @ Qtb.X  # (bs, n, dx_out)
        probE = dense_data.E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        if dense_data.charges.numel() > 0:
            prob_charges = dense_data.charges @ Qtb.charges
        else:
            prob_charges = dense_data.charges

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX,
            probE=probE,
            prob_charges=prob_charges,
            node_mask=dense_data.node_mask,
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.X_classes)
        E_t = F.one_hot(sampled_t.E, num_classes=self.E_classes)
        assert (dense_data.X.shape == X_t.shape) and (dense_data.E.shape == E_t.shape)
        if sampled_t.charges.numel() > 0:
            charges_t = F.one_hot(sampled_t.charges, num_classes=self.charges_classes)
        else:
            charges_t = X_t.new_zeros((*X_t.shape[:-1], 0))

        z_t = utils.PlaceHolder(
            X=X_t,
            charges=charges_t,
            E=E_t,
            y=dense_data.y,
            t_int=t_int,
            t=t_float,
            node_mask=dense_data.node_mask,
        ).mask()
        return z_t

    def get_limit_dist(self):
        X_marginals = self.X_marginals + 1e-7
        X_marginals = X_marginals / torch.sum(X_marginals)
        E_marginals = self.E_marginals + 1e-7
        E_marginals = E_marginals / torch.sum(E_marginals)
        charges_marginals = self.charges_marginals + 1e-7
        charges_marginals = charges_marginals / torch.sum(charges_marginals)
        limit_dist = utils.PlaceHolder(
            X=X_marginals, E=E_marginals, charges=charges_marginals, y=None
        )

        return limit_dist

    def sample_limit_dist(self, node_mask):
        """Sample from the limit distribution of the diffusion process"""

        def sample_from_categorical(probs_tensor):
            """Defining this function because torch.multinomial sometimes samples types that have 0 probability. This is problematic for the case of absorbing transition matrices, where all the edges should be 'no-edge type' and it was not happening."""
            if torch.nonzero(probs_tensor[:, 1:]).numel() == 0:
                # if all the probabilities are 0 except the first type, then sample from the first type
                sampled = torch.zeros(probs_tensor.shape[0], 1, dtype=torch.int64)
                # parse to int because entries are indexes
            else:
                sampled = probs_tensor.multinomial(1)
            return sampled

        bs, n_max = node_mask.shape
        x_limit = self.X_marginals.expand(bs, n_max, -1)
        e_limit = self.E_marginals[None, None, None, :].expand(bs, n_max, n_max, -1)
        charges_limit = self.charges_marginals.expand(bs, n_max, -1)

        U_X = (
            sample_from_categorical(x_limit.flatten(end_dim=-2))
            .reshape(bs, n_max)
            .to(node_mask.device)
        )

        U_E = (
            sample_from_categorical(e_limit.flatten(end_dim=-2))
            .reshape(bs, n_max, n_max)
            .to(node_mask.device)
        )
        U_y = torch.zeros((bs, 0), device=node_mask.device)

        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

        if charges_limit.numel() > 0:
            U_c = (
                sample_from_categorical(charges_limit.flatten(end_dim=-2))
                .reshape(bs, n_max)
                .to(node_mask.device)
            )
            U_c = F.one_hot(U_c, num_classes=charges_limit.shape[-1]).float()
        else:
            U_c = U_X.new_zeros((*U_X.shape[:-1], 0))

        # Get upper triangular part of edge noise, without main diagonal
        upper_triangular_mask = torch.zeros_like(U_E)
        indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
        upper_triangular_mask[:, indices[0], indices[1], :] = 1

        U_E = U_E * upper_triangular_mask
        U_E = U_E + torch.transpose(U_E, 1, 2)
        assert (U_E == torch.transpose(U_E, 1, 2)).all()

        t_array = node_mask.new_ones((node_mask.shape[0], 1))
        t_int_array = self.T * t_array.long()
        return utils.PlaceHolder(
            X=U_X,
            charges=U_c,
            E=U_E,
            y=U_y,
            t_int=t_int_array,
            t=t_array,
            node_mask=node_mask,
        ).mask(node_mask)

    def sample_zs_from_zt_and_pred(self, z_t, pred, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        bs, n, dxs = z_t.X.shape
        node_mask = z_t.node_mask
        t_int = z_t.t_int

        # Retrieve transitions matrix
        Qtb = self.get_Qt_bar(t_int=t_int)
        Qsb = self.get_Qt_bar(t_int=s_int)
        Qt = self.get_Qt(t_int)

        # Normalize predictions for the categorical features
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        pred_charges = F.softmax(pred.charges, dim=-1)

        p_s_and_t_given_0_X = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=z_t.X, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
        )

        p_s_and_t_given_0_E = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=z_t.E, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
            )
        )
        p_s_and_t_given_0_c = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=z_t.charges, Qt=Qt.charges, Qsb=Qsb.charges, Qtb=Qtb.charges
            )
        )

        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # bs, n, d_t-1

        weighted_c = (
            pred_charges.unsqueeze(-1) * p_s_and_t_given_0_c
        )  # bs, n, d0, d_t-1
        unnormalized_prob_c = weighted_c.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_c[torch.sum(unnormalized_prob_c, dim=-1) == 0] = 1e-5
        prob_c = unnormalized_prob_c / torch.sum(
            unnormalized_prob_c, dim=-1, keepdim=True
        )  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        if prob_c.numel() > 0:
            assert ((prob_c.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, prob_c, node_mask=z_t.node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=self.X_classes).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.E_classes).float()

        if prob_c.numel() > 0:
            charges_s = F.one_hot(
                sampled_s.charges, num_classes=self.charges_classes
            ).float()
        else:
            charges_s = X_s.new_zeros((bs, n, 0))

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (z_t.X.shape == X_s.shape) and (z_t.E.shape == E_s.shape)

        z_s = utils.PlaceHolder(
            X=X_s,
            charges=charges_s,
            E=E_s,
            y=torch.zeros(z_t.y.shape[0], 0, device=X_s.device),
            t_int=s_int,
            t=s_int / self.T,
            node_mask=node_mask,
        ).mask(node_mask)

        return z_s


class DiscreteUniformTransition(NoiseModel):
    def __init__(self, cfg, output_dims):
        super().__init__(cfg=cfg)
        self.X_classes = output_dims.X
        self.charges_classes = output_dims.charges
        self.E_classes = output_dims.E
        self.y_classes = output_dims.y
        self.X_marginals = torch.ones(self.X_classes) / self.X_classes
        self.charges_marginals = torch.ones(self.charges_classes) / self.charges_classes
        self.E_marginals = torch.ones(self.E_classes) / self.E_classes
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes
        super().complete_init()


class MarginalTransition(NoiseModel):
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes):
        super().__init__(cfg=cfg)
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.charges_classes = len(charges_marginals)
        self.y_classes = y_classes
        self.X_marginals = x_marginals
        self.E_marginals = e_marginals
        self.charges_marginals = charges_marginals
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes
        super().complete_init()


class AbsorbingTransition(NoiseModel):
    def __init__(self, cfg, output_dims):
        super().__init__(cfg)
        self.X_classes = output_dims.X
        self.charges_classes = output_dims.charges
        self.E_classes = output_dims.E
        self.y_classes = output_dims.y
        self.X_marginals = torch.zeros(self.X_classes)
        self.X_marginals[0] = 1
        self.E_marginals = torch.zeros(self.E_classes)
        self.E_marginals[0] = 1
        self.charges_marginals = torch.zeros(self.charges_classes)
        if self.charges_classes > 0:
            self.charges_marginals[0] = 1
        self.y_marginals = torch.ones(self.y_classes) / self.y_classes
        super().complete_init()


class PlanarTransition(MarginalTransition):
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes):
        super().__init__(cfg, x_marginals, e_marginals, charges_marginals, y_classes)
        # Absorbing only on edges
        self.E_marginals = torch.zeros(self.E_marginals.shape)
        self.E_marginals[0] = 1
        super().complete_init()

        # Schedule for absorbing scheme is different
        betas_abs = diffusion_utils.linear_beta_schedule(self.timesteps, self.nu_arr)
        self._betas_abs = torch.from_numpy(betas_abs)
        self._alphas_abs = 1 - torch.clamp(self._betas_abs, min=0, max=0.9999)
        self._alphas_abs_bar = torch.cumprod(self._alphas_abs, dim=0)
        # Marginals not counting with absorbing (no-edge state)
        e_real_marginals = e_marginals[1:] / torch.sum(e_marginals[1:])
        self.Pe_real = e_real_marginals.unsqueeze(0).expand(self.E_classes - 1, -1)

        # Plot absorbing alphas and betas
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(self._betas_abs[:, 0], label="beta_abs")
        # plt.plot(self._alphas_abs[:, 0], label="alpha_abs")
        # plt.plot(self._alphas_abs_bar[:, 0], label="alpha_abs_bar")
        # plt.legend()
        # plt.savefig("absorbing_schedule.png")

    def get_beta_abs(self, t_normalized=None, t_int=None, key=None):
        return self._get(
            self._betas_abs, t_normalized=t_normalized, t_int=t_int, key=key
        )

    def get_alpha_abs_bar(self, t_normalized=None, t_int=None, key=None):
        return self._get(
            self._alphas_abs_bar, t_normalized=t_normalized, t_int=t_int, key=key
        )

    def get_Qt(self, t_int):
        Qt = super().get_Qt(t_int=t_int)

        # Planar specificities only on edges, the rest is inherited from marginal
        dev = t_int.device
        Pe_real = self.Pe_real.float().to(dev)
        be = self.get_beta(t_int=t_int, key="e").unsqueeze(1)
        be_abs = self.get_beta_abs(t_int=t_int, key="e").unsqueeze(1)
        bs = be_abs.shape[0]
        q_e = torch.eye(self.E_classes, device=dev).expand(bs, -1, -1).clone()
        q_e[:, 1:, 0] = be_abs.expand(-1, self.E_classes - 1, -1).squeeze(dim=-1)
        q_e[:, 1:, 1:] = (1 - be_abs) * (
            (1 - be) * torch.eye(self.E_classes - 1, device=dev).unsqueeze(0)
            + be * Pe_real
        )

        return utils.PlaceHolder(X=Qt.X, charges=Qt.charges, E=q_e, y=Qt.y)

    def get_Qt_bar(self, t_int):
        Qt_bar = super().get_Qt_bar(t_int=t_int)

        # Planar specificities only on edges, the rest is inherited from marginal
        dev = t_int.device
        Pe_real = self.Pe_real.float().to(dev)
        a_e = self.get_alpha_bar(t_int=t_int, key="e").unsqueeze(1)
        a_e_abs = self.get_alpha_abs_bar(t_int=t_int, key="e").unsqueeze(1)
        bs = a_e_abs.shape[0]
        q_e = torch.eye(self.E_classes, device=dev).expand(bs, -1, -1).clone()
        # expand() only creates a new view on the existing tensor, thus it doesn't allocate the full memory necessary to receive all the elements from the following operations, which have more elements than the original tensor. by cloning the tensor, we allocate the full memory necessary.
        q_e[:, 1:, 0] = (1 - a_e_abs).expand(-1, self.E_classes - 1, -1).squeeze(dim=-1)
        q_e[:, 1:, 1:] = a_e_abs * (
            a_e * torch.eye(self.E_classes - 1, device=dev).unsqueeze(0)
            + (1 - a_e) * Pe_real
        )

        # Ensure that each row of transition matrix sums to 1
        assert ((q_e.sum(dim=2) - 1.0).abs() < 1e-4).all()

        return utils.PlaceHolder(X=Qt_bar.X, charges=Qt_bar.charges, E=q_e, y=Qt_bar.y)


class AbsorbingEdgesTransition(MarginalTransition):
    def __init__(self, cfg, x_marginals, e_marginals, charges_marginals, y_classes):
        super().__init__(cfg, x_marginals, e_marginals, charges_marginals, y_classes)
        # Absorbing only on edges
        self.E_marginals = torch.zeros(self.E_marginals.shape)
        self.E_marginals[0] = 1
        super().complete_init()

        # Schedule for absorbing scheme is different
        betas_abs = diffusion_utils.linear_beta_schedule(self.timesteps, self.nu_arr)
        self._betas_abs = torch.from_numpy(betas_abs)
        self._alphas_abs = 1 - torch.clamp(self._betas_abs, min=0, max=0.9999)
        self._alphas_abs_bar = torch.cumprod(self._alphas_abs, dim=0)

    def get_beta_abs(self, t_normalized=None, t_int=None, key=None):
        return self._get(
            self._betas_abs, t_normalized=t_normalized, t_int=t_int, key=key
        )

    def get_alpha_abs_bar(self, t_normalized=None, t_int=None, key=None):
        return self._get(
            self._alphas_abs_bar, t_normalized=t_normalized, t_int=t_int, key=key
        )

    def get_Qt(self, t_int):
        Qt = super().get_Qt(t_int)

        dev = t_int.device
        Pe = self.Pe.float().to(dev)
        be_abs = self.get_beta_abs(t_int=t_int, key="e").unsqueeze(1)
        q_e = be_abs * Pe + (1 - be_abs) * torch.eye(
            self.E_classes, device=dev
        ).unsqueeze(0)

        return utils.PlaceHolder(X=Qt.X, charges=Qt.charges, E=q_e, y=Qt.y)

    def get_Qt_bar(self, t_int):
        Qt_bar = super().get_Qt_bar(t_int)

        dev = t_int.device
        Pe = self.Pe.float().to(dev)
        a_e_abs = self.get_alpha_abs_bar(t_int=t_int, key="e").unsqueeze(1)
        q_e = (
            a_e_abs * torch.eye(self.E_classes, device=dev).unsqueeze(0)
            + (1 - a_e_abs) * Pe
        )

        assert ((q_e.sum(dim=2) - 1.0).abs() < 1e-4).all()
        return utils.PlaceHolder(X=Qt_bar.X, charges=Qt_bar.charges, E=q_e, y=Qt_bar.y)
