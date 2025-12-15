import torch
from typing import Optional, Dict, Any

# ============================================================
#  Forward projector: y = A x
#  - Exact MBIRcone separable model
#  - Optimized inner loops (v, w, z)
# ============================================================
def forward_project_fast(
    x: torch.Tensor,
    A: Dict[str, torch.Tensor],
    img_params: Any,
    sino_params: Any,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> torch.Tensor:

    # ----------------------------
    # Device handling
    # ----------------------------
    if device is None:
        device = x.device

    # Image / sino sizes
    Nz, Ny, Nx = x.shape
    N_beta = sino_params.N_beta
    N_dv   = sino_params.N_dv
    N_dw   = sino_params.N_dw

    # Allocate output sinogram
    Ax = torch.zeros((N_beta, N_dv, N_dw), dtype=torch.float32, device=device)

    # ----------------------------
    # Unpack A and move to device
    # ----------------------------
    B = A["B"].to(device).contiguous()
    C = A["C"].to(device).contiguous()
    j_u = A["j_u"].to(device).contiguous()

    i_vstart      = A["i_vstart"].to(device).contiguous()
    i_vstride     = A["i_vstride"].to(device).contiguous()
    i_vstride_max = int(A["i_vstride_max"])

    i_wstart      = A["i_wstart"].to(device).contiguous()
    i_wstride     = A["i_wstride"].to(device).contiguous()
    i_wstride_max = int(A["i_wstride_max"])

    B_ij_scaler = float(A["B_ij_scaler"])
    C_ij_scaler = float(A["C_ij_scaler"])

    # x in C is indexed as x[j_x, j_y, j_z]
    # our x is (Nz, Ny, Nx)
    x_vol = x.to(device).permute(2, 1, 0).contiguous()  # (Nx, Ny, Nz)

    # Cache for iw ranges
    iw_range_cache: Dict[tuple, torch.Tensor] = {}

    # =========================================================
    # Main loops
    # =========================================================
    beta_iter = range(N_beta)
    for i_beta in beta_iter:

        ju_grid = j_u[:, :, i_beta]          # (Nx, Ny)
        iv0     = i_vstart[:, :, i_beta]     # (Nx, Ny)
        ivlen   = i_vstride[:, :, i_beta]    # (Nx, Ny)

        base_B_idx = i_beta * i_vstride_max

        # ----------------------------
        # Mask valid (j_x, j_y)
        # ----------------------------
        xy_mask = (ju_grid >= 0) & (ivlen > 0)
        valid_jx, valid_jy = torch.where(xy_mask)

        if valid_jx.numel() == 0:
            continue

        for idx in range(len(valid_jx)):
            j_x = valid_jx[idx]
            j_y = valid_jy[idx]

            ju = ju_grid[j_x, j_y]
            v0 = iv0[j_x, j_y]
            vl = ivlen[j_x, j_y]

            # -------------------------
            # Vectorized V-direction
            # -------------------------
            iv_range = torch.arange(v0, v0 + vl, device=device)
            Bij_idx = base_B_idx + (iv_range - v0)
            B_vec = B_ij_scaler * B[j_x, j_y, Bij_idx]

            if torch.all(B_vec == 0):
                continue

            # -------------------------
            # Z-direction (VECTORIZED)
            # -------------------------
            x_line = x_vol[j_x, j_y, :]  # (Nz)

            iw0_line = i_wstart[ju]
            iwlen_line = i_wstride[ju]

            z_mask = (x_line != 0) & (iwlen_line > 0)
            Z = torch.where(z_mask)[0]  # (Nz_valid,)

            if Z.numel() == 0:
                continue

            # Gather x(z)
            Xz = x_line[Z]  # (Nz_valid,)

            # Assume all valid z share same w-range pattern PER ju
            # (this matches MBIRcone geometry)
            w0 = iw0_line[Z[0]]
            wl = iwlen_line[Z[0]]

            key = (w0.item(), wl.item())
            if key not in iw_range_cache:
                iw_range_cache[key] = torch.arange(w0, w0 + wl, device=device)
            iw_range = iw_range_cache[key]  # (wl,)

            # Build C-matrix: (Nz_valid, wl)
            C_idx = Z[:, None] * i_wstride_max + (iw_range[None, :] - w0)
            C_mat = C_ij_scaler * C[ju, C_idx]

            # ✅ EARLY EXIT — best place
            if torch.all(C_mat == 0):
                continue

            # Sum over Z:  S(w) = Σ_z x(z) * C(z,w)
            S = (Xz[:, None] * C_mat).sum(dim=0)  # (wl,)

            # Outer product over V × W
            Ax[i_beta, iv_range[:, None], iw_range[None, :]] += (
                    B_vec[:, None] * S[None, :]
            )

    return Ax


# ============================================================
#  Back projector: x = A^T y
#  - Exact adjoint of forward_project_fast
#  - Masked XY + Z loops
# ============================================================
def back_project_fast(
    Ax: torch.Tensor,
    A: Dict[str, torch.Tensor],
    img_params: Any,
    sino_params: Any,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> torch.Tensor:

    if device is None:
        device = Ax.device

    Ax = Ax.to(device).contiguous()

    N_beta = sino_params.N_beta
    Nx = img_params.N_x
    Ny = img_params.N_y
    Nz = img_params.N_z

    # Output volume
    x_vol = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=device)

    # Unpack A
    B = A["B"].to(device).contiguous()
    C = A["C"].to(device).contiguous()
    j_u = A["j_u"].to(device).contiguous()

    i_vstart      = A["i_vstart"].to(device).contiguous()
    i_vstride     = A["i_vstride"].to(device).contiguous()
    i_vstride_max = int(A["i_vstride_max"])

    i_wstart      = A["i_wstart"].to(device).contiguous()
    i_wstride     = A["i_wstride"].to(device).contiguous()
    i_wstride_max = int(A["i_wstride_max"])

    B_ij_scaler = float(A["B_ij_scaler"])
    C_ij_scaler = float(A["C_ij_scaler"])

    iw_range_cache: Dict[tuple, torch.Tensor] = {}

    beta_iter = range(N_beta)

    for i_beta in beta_iter:

        ju_grid = j_u[:, :, i_beta]
        iv0     = i_vstart[:, :, i_beta]
        ivlen   = i_vstride[:, :, i_beta]

        base_B_idx = i_beta * i_vstride_max

        xy_mask = (ju_grid >= 0) & (ivlen > 0)
        valid_jx, valid_jy = torch.where(xy_mask)

        if valid_jx.numel() == 0:
            continue

        for idx in range(valid_jx.numel()):

            j_x = valid_jx[idx]
            j_y = valid_jy[idx]

            ju = ju_grid[j_x, j_y]
            v0 = iv0[j_x, j_y]
            vl = ivlen[j_x, j_y]

            iv_range = torch.arange(v0, v0 + vl, device=device)
            Bij_idx = base_B_idx + (iv_range - v0)
            B_vec = B_ij_scaler * B[j_x, j_y, Bij_idx]

            if torch.all(B_vec == 0):
                continue

            iw0_line   = i_wstart[ju]
            iwlen_line = i_wstride[ju]

            # ----------------------------
            # Z-direction (vectorized)
            # ----------------------------
            valid_z = torch.where(iwlen_line > 0)[0]
            if valid_z.numel() == 0:
                continue

            # Same w-range for all z (MBIRcone assumption)
            j_z0 = valid_z[0]
            w0 = iw0_line[j_z0]
            wl = iwlen_line[j_z0]

            key = (w0.item(), wl.item())
            if key not in iw_range_cache:
                iw_range_cache[key] = torch.arange(w0, w0 + wl, device=device)
            iw_range = iw_range_cache[key]

            # Extract sinogram block once
            sino_block = Ax[
                i_beta,
                iv_range[:, None],
                iw_range[None, :]
            ]  # (vl, wl)

            # Sum over v
            T_w = (B_vec[:, None] * sino_block).sum(dim=0)  # (wl,)

            # Build C matrix
            C_idx_base = valid_z * i_wstride_max
            w_offset = torch.arange(wl, device=device)
            C_idx_mat = C_idx_base[:, None] + w_offset[None, :]

            C_mat = C_ij_scaler * C[ju, C_idx_mat]  # (Nz_valid, wl)

            if torch.all(C_mat == 0):
                continue

            # Sum over w
            contrib = (C_mat * T_w[None, :]).sum(dim=1)  # (Nz_valid,)

            x_vol[j_x, j_y, valid_z] += contrib

    return x_vol.permute(2, 1, 0).contiguous()



# ============================================================
#  Autograd wrapper (for PyTorch optimization / MBIR)
# ============================================================
class ConeBeamProjectorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, img_params, sino_params, device=None):
        # x is (Nz, Ny, Nx)
        Ax = forward_project_fast(
            x, A, img_params, sino_params, device=device, show_progress=False
        )
        ctx.A = A
        ctx.img_params = img_params
        ctx.sino_params = sino_params
        ctx.device = device if device is not None else x.device
        return Ax

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output has shape (N_beta, N_dv, N_dw)
        grad_x = back_project_fast(
            grad_output,
            ctx.A,
            ctx.img_params,
            ctx.sino_params,
            device=ctx.device,
            show_progress=False,
        )
        # Only x has gradients; A, params are non-differentiable here
        return grad_x, None, None, None, None


def cone_beam_projector(x, A, img_params, sino_params, device=None):
    """
    Convenience wrapper that behaves like a normal PyTorch op:
        Ax = cone_beam_projector(x, A, img_params, sino_params)
    """
    return ConeBeamProjectorFn.apply(x, A, img_params, sino_params, device)


