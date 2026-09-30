from typing import *
import torch
from .. import SparseTensor
from .. import DEBUG, ATTN

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_scaled_dot_product_attention',
    'sparse_block_diagonal_self_attention',
    'sparse_region_boost_self_attention',
]


def sparse_block_diagonal_self_attention(qkv: SparseTensor, group_ids: torch.Tensor) -> SparseTensor:
    """Full self-attention restricted so each voxel only attends within its own (batch, group).

    Used for mixed text/image conditioning: voxels of one modality's region are made blind to the
    other region's voxels, so a strongly-conditioned region cannot bleed into a cfg=0 region through
    global self-attention. Implemented as block-diagonal attention over the packed sequence (via
    variable-length sequences), which is memory-safe (no N x N mask) and cheaper than full attention.

    Args:
        qkv (SparseTensor): [N, 3, H, C] packed q/k/v (RoPE/rms already applied by the caller).
        group_ids (torch.Tensor): [T] long per-voxel group id, aligned to qkv.feats rows.
    """
    feats = qkv.feats  # [T, 3, H, C]
    T = feats.shape[0]
    device = feats.device
    if T == 0:
        return qkv.replace(feats[:, 0])

    # Block key = (batch index, group id); different batch items or groups never attend together.
    batch_idx = qkv.coords[:, 0].long()
    num_groups = int(group_ids.max().item()) + 1
    block_key = batch_idx * num_groups + group_ids.long()

    # Reorder so each block is contiguous (order within a block is irrelevant given baked RoPE).
    perm = torch.argsort(block_key)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(T, device=device)
    feats_sorted = feats[perm]
    keys_sorted = block_key[perm]
    _, counts = torch.unique_consecutive(keys_sorted, return_counts=True)

    if ATTN == 'flash_attn':
        cu_seqlens = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
        out_sorted = flash_attn.flash_attn_varlen_qkvpacked_func(
            feats_sorted, cu_seqlens, int(counts.max().item()))
    elif ATTN == 'xformers':
        q, k, v = feats_sorted.unbind(dim=1)  # each [T, H, C]
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(counts.tolist())
        out_sorted = xops.memory_efficient_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask)[0]
    else:
        raise ValueError(f"Unknown attention module: {ATTN}")

    out = out_sorted[inv]
    return qkv.replace(out)


def _lse_to_token_major(lse: torch.Tensor, tokens: int, heads: int) -> torch.Tensor:
    """Normalize flash-attn's varlen log-sum-exp to [T, H, 1] float.

    It comes back as [heads, tokens], and some builds pad the token axis out to a multiple
    of the kernel's block size, so trim to the real token count after transposing.
    """
    if lse.shape[0] == heads and lse.shape[0] != tokens:
        lse = lse.transpose(0, 1)
    return lse[:tokens].unsqueeze(-1).float()


def sparse_region_boost_self_attention(qkv: SparseTensor, group_ids: torch.Tensor, boost: float) -> SparseTensor:
    """Full self-attention with in-region attention weights multiplied by `boost`.

    The soft counterpart of :func:`sparse_block_diagonal_self_attention`: a voxel still sees the
    whole object, so parts stay globally coherent, but it weighs its own region `boost` times
    more heavily. That is what keeps one condition's identity from spilling into a region that is
    unguided (cfg=0) on the current step of alternating text/image conditioning, without the
    independently-generated look that hard blocking produces.

    Multiplying the in-region weights by `b` is exactly adding `log b` to the in-region logits
    before the softmax. Materializing that bias would need an O(T^2) score matrix in every block
    of every step; it is not necessary, because the result is recoverable from two flash calls
    and their log-sum-exps:

        out = (b*Z_in*out_in + (Z_all*out_all - Z_in*out_in)) / (b*Z_in + (Z_all - Z_in))

    where `Z = exp(lse)` is each softmax's unnormalised mass and `Z_in <= Z_all` always.

    Args:
        qkv (SparseTensor): [N, 3, H, C] packed q/k/v (RoPE/rms already applied by the caller).
        group_ids (torch.Tensor): [T] long per-voxel region id, aligned to qkv.feats rows.
        boost (float): in-region weight multiplier; 1.0 is plain full attention.
    """
    feats = qkv.feats  # [T, 3, H, C]
    T = feats.shape[0]
    device = feats.device
    if T == 0:
        return qkv.replace(feats[:, 0])
    if boost == 1.0:
        return sparse_scaled_dot_product_attention(qkv)
    if ATTN != 'flash_attn':
        raise NotImplementedError(
            f"region-boosted self-attention needs the flash_attn backend for its log-sum-exp, "
            f"but the sparse attention backend is {ATTN!r}. Set ATTN=flash_attn, or disable the "
            f"boost (sim_guidance.mixed_self_attn_boost: 1.0) to fall back to stock attention."
        )

    # Block key = (batch index, region id), so sorting makes each region contiguous AND, because
    # batch is the major key, keeps each batch item contiguous too - both segmentations below
    # read straight off the same permutation.
    batch_idx = qkv.coords[:, 0].long()
    num_groups = int(group_ids.max().item()) + 1
    block_key = batch_idx * num_groups + group_ids.long()

    perm = torch.argsort(block_key)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(T, device=device)
    feats_sorted = feats[perm].contiguous()
    heads = feats_sorted.shape[2]

    # Restricted softmax: each region attends only within itself.
    _, region_counts = torch.unique_consecutive(block_key[perm], return_counts=True)
    cu_in = torch.cat([region_counts.new_zeros(1), region_counts.cumsum(0)]).to(torch.int32)
    out_in, lse_in, _ = flash_attn.flash_attn_varlen_qkvpacked_func(
        feats_sorted, cu_in, int(region_counts.max().item()), return_attn_probs=True)

    # Unrestricted softmax over the same permuted layout: blocks are whole batch items.
    _, batch_counts = torch.unique_consecutive(batch_idx[perm], return_counts=True)
    cu_all = torch.cat([batch_counts.new_zeros(1), batch_counts.cumsum(0)]).to(torch.int32)
    out_all, lse_all, _ = flash_attn.flash_attn_varlen_qkvpacked_func(
        feats_sorted, cu_all, int(batch_counts.max().item()), return_attn_probs=True)

    # Work relative to lse_all for numerical stability; Z_in / Z_all lands in (0, 1].
    li = _lse_to_token_major(lse_in, T, heads)
    la = _lse_to_token_major(lse_all, T, heads)
    w_in = torch.exp(li - la)
    numer = (boost - 1.0) * w_in * out_in.float() + out_all.float()
    denom = (boost - 1.0) * w_in + 1.0
    out_sorted = (numer / denom).to(feats.dtype)
    return qkv.replace(out_sorted[inv])


@overload
def sparse_scaled_dot_product_attention(qkv: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (SparseTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, kv: Union[SparseTensor, torch.Tensor]) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, L, H, C] dense tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: SparseTensor, v: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: torch.Tensor, v: torch.Tensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: SparseTensor, v: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, SparseTensor) and isinstance(kv, (SparseTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, SparseTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, SparseTensor) and isinstance(k, (SparseTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, SparseTensor) and isinstance(v, SparseTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if DEBUG:
        if s is not None:
            for i in range(s.shape[0]):
                assert (s.coords[s.layout[i]] == i).all(), f"SparseScaledDotProductSelfAttention: batch index mismatch"
        if num_all_args in [2, 3]:
            assert q.shape[:2] == [1, sum(q_seqlen)], f"SparseScaledDotProductSelfAttention: q shape mismatch"
        if num_all_args == 3:
            assert k.shape[:2] == [1, sum(kv_seqlen)], f"SparseScaledDotProductSelfAttention: k shape mismatch"
            assert v.shape[:2] == [1, sum(kv_seqlen)], f"SparseScaledDotProductSelfAttention: v shape mismatch"

    if ATTN == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif ATTN == 'flash_attn':
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    else:
        raise ValueError(f"Unknown attention module: {ATTN}")
    
    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
