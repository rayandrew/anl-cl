# flake8: noqa: E501
from src.helpers.config import Config
from src.helpers.definitions import Dataset
from src.transforms.base import BaseFeatureEngineering

NO_FEATS = "no-feature"


def get_features(config: Config) -> BaseFeatureEngineering:
    feature = config.dataset.feature
    dataset_name = config.dataset.name

    if dataset_name == Dataset.AZURE:
        if feature == NO_FEATS:
            from src.transforms.azure_vmcpu import NoFeats as AzureNoFeats

            return AzureNoFeats(config)
        elif feature == "feature-a":
            from src.transforms.azure_vmcpu import (
                FeatureEngineering_A as AzureFeatureEngineering_A,
            )

            return AzureFeatureEngineering_A(config)
        elif feature == "feature-b":
            from src.transforms.azure_vmcpu import (
                FeatureEngineering_B as AzureFeatureEngineering_B,
            )

            return AzureFeatureEngineering_B(config)
        elif feature == "feature-c":
            from src.transforms.azure_vmcpu import (
                FeatureEngineering_C as AzureFeatureEngineering_C,
            )

            return AzureFeatureEngineering_C(config)
        else:
            raise ValueError(f"[AZURE] Unknown feature: {feature}")
    elif dataset_name == Dataset.GOOGLE:
        if feature == NO_FEATS:
            from src.transforms.google_scheduler import (
                FeatureEngineering_Baseline as GoogleFeatureEngineering_Baseline,
            )

            return GoogleFeatureEngineering_Baseline(config)
        elif feature == "feature-a":
            from src.transforms.google_scheduler import (
                FeatureEngineering_A as GoogleFeatureEngineering_A,
            )

            return GoogleFeatureEngineering_A(config)
        elif feature == "feature-b":
            from src.transforms.google_scheduler import (
                FeatureEngineering_B as GoogleFeatureEngineering_B,
            )

            return GoogleFeatureEngineering_B(config)
        else:
            raise ValueError(f"[GOOGLE] Unknown feature: {feature}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


__all__ = ["get_features", "NO_FEATS"]
