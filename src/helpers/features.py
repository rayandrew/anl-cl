from src.helpers.config import Config
from src.helpers.definitions import Dataset
from src.transforms.base import BaseFeatureTransformSet

NO_FEATS = "no-feature"


def get_features(config: Config) -> BaseFeatureTransformSet:
    feature = config.dataset.feature
    dataset_name = config.dataset.name

    if dataset_name == Dataset.AZURE:
        if feature == NO_FEATS:
            from src.transforms.azure_vmcpu import NoFeats

            return NoFeats(config)
        elif feature == "feats-a":
            from src.transforms.azure_vmcpu import FeatureA_TransformSet

            return FeatureA_TransformSet(config)
        else:
            raise ValueError(f"[AZURE] Unknown feature: {feature}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = ["get_features", "NO_FEATS"]
