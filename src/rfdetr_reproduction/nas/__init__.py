"""Weight-Sharing NAS search space, validation, sampling, and official RF-DETR adapter."""

from rfdetr_reproduction.nas.sampling import (
    get_train_cfg,
    sample_biased_subnet,
    sample_uniform_subnet,
)
from rfdetr_reproduction.nas.search_space import (
    build_infer_search_space_7ps,
    build_infer_search_space_8ps,
    build_representative_infer_configs,
    build_train_search_space,
    build_validated_search_space,
)
from rfdetr_reproduction.nas.validator import validate_nas_config

__all__ = [
    "validate_nas_config",
    "build_validated_search_space",
    "build_train_search_space",
    "build_infer_search_space_8ps",
    "build_infer_search_space_7ps",
    "build_representative_infer_configs",
    "get_train_cfg",
    "sample_uniform_subnet",
    "sample_biased_subnet",
]
