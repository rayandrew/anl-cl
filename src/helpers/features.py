from src.helpers.config import Config
from src.helpers.definitions import Dataset
from src.transforms.base import BaseFeatureEngineering

NO_FEATS = "no-feature"


def get_features(config: Config) -> BaseFeatureEngineering:
    feature = config.dataset.feature
    dataset_name = config.dataset.name

    if dataset_name == Dataset.AZURE:
        if feature == NO_FEATS:
            from src.transforms.azure_vmcpu import NoFeats

            return NoFeats(config)
        elif feature == "feature-a":
            from src.transforms.azure_vmcpu import FeatureEngineering_A

            return FeatureEngineering_A(config)
        else:
            raise ValueError(f"[AZURE] Unknown feature: {feature}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = ["get_features", "NO_FEATS"]
