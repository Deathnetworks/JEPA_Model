import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Attempt to load the SYCL C++ extension if compiled
    from sycl_mamba import selective_scan
    USE_SYCL_KERNEL = True
except ImportError:
    USE_SYCL_KERNEL = False

class OptimizedMambaBlock(nn.Module):
    """
    Optimized State Space Duality (SSD) Layer wrapper.
    Attempts to use the native SYCL kernel for Intel Arc Pro (XPU),
    with a fallback to a legacy Python simulation or mamba-ssm.
    """
    def __init__(self, d_model=6144, d_state=256, nheads=128, use_sycl_kernel=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.nheads = nheads

        self.d_head = 64
        self.d_inner = nheads * self.d_head

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=4, padding=3, groups=self.d_inner
        )
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Determine if we can and should use the SYCL kernel
        self.use_sycl_kernel = use_sycl_kernel and USE_SYCL_KERNEL and hasattr(torch, 'xpu') and torch.xpu.is_available()

    def forward(self, x, prev_state=None):
        batch, seq_len, _ = x.shape
        proj = self.in_proj(x)

        x_inner, z, B, C, dt = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads],
            dim=-1
        )

        x_inner = x_inner.transpose(1, 2)
        x_inner = F.silu(self.conv1d(x_inner)[:, :, :seq_len])
        x_inner = x_inner.transpose(1, 2)

        z = F.silu(z)

        if self.use_sycl_kernel:
            # Assuming A is a static parameter here for the kernel signature
            A = torch.ones((self.nheads,), device=x.device, dtype=x.dtype)
            out = selective_scan(x_inner, dt, A, B, C)
        else:
            # Legacy fallback loop (Simulated recurrence)
            h = torch.zeros(batch, self.d_inner, device=x.device, dtype=x.dtype)
            if prev_state is not None:
                h = prev_state

            out = torch.zeros_like(x_inner)
            dt = F.softplus(dt)

            # Simple simulation of Mamba recurrence
            for t in range(seq_len):
                dt_t = dt[:, t, :].unsqueeze(-1).expand(-1, -1, self.d_head).reshape(batch, self.d_inner)
                B_t = B[:, t, :]
                C_t = C[:, t, :]
                x_t = x_inner[:, t, :]

                h = h * torch.exp(-dt_t) + x_t
                out[:, t, :] = h

        out = out * z
        return self.out_proj(out), h
