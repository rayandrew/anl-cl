from attrs import define, field, fields

from src.config.dataset import (
    AlibabaMachineNonSequenceDataset,
    AlibabaMachineSequenceDataset,
)


@define
class Config:
    dataset: AlibabaMachineSequenceDataset | AlibabaMachineNonSequenceDataset = (
        field()
    )


__all__ = ["Config"]
