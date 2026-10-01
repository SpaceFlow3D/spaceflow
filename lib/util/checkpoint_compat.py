"""Scoped compatibility for the recorded upstream PartField checkpoint."""

from contextlib import contextmanager


@contextmanager
def partfield_checkpoint_scope():
    """Allow the checkpoint's YACS configuration with PyTorch weights-only loading.

    PartField stores a CfgNode alongside its tensors. Lightning 2.2 delegates
    loading to torch.load, whose default changed in PyTorch 2.6. Keep that
    default and allow only this configuration class while PartField loads.
    """
    import torch
    from yacs.config import CfgNode

    with torch.serialization.safe_globals([CfgNode]):
        yield
