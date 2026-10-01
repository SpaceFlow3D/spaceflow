"""Conditioning compatibility checks without importing GPU dependencies."""


def can_reuse_appearance_pipeline(pipeline, appearance_type):
    """Image appearance needs the image pipeline's preprocessing interface."""
    if pipeline is None:
        return False
    if appearance_type == "image":
        return callable(getattr(pipeline, "preprocess_image", None))
    return True
