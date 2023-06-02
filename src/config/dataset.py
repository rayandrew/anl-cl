from attrs import define, field, fields
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@define(slots=False)
class NonSequenceDataset:
    target: str
    name: str
    path: str


@define(slots=False)
class SequenceDataset(NonSequenceDataset):
    seq_len: int = field(default=5)
    univariate: bool = False

    @seq_len.validator
    def _check_seq_len(self, attribute, value):
        if value <= 1:
            raise ValueError(
                "seq_len must be positive and greater than 1"
            )


@define(slots=False)
class AlibabaMachineNonSequenceDataset(NonSequenceDataset):
    name: (
        fields(NonSequenceDataset).target.evolve(
            default="alibaba_machine_non_seq"
        )
    )
    target: str = fields(NonSequenceDataset).target.evolve(
        default="src.dataset.AlibabaMachineDataset"
    )


@define(slots=False)
class AlibabaMachineSequenceDataset(SequenceDataset):
    name: (
        fields(NonSequenceDataset).target.evolve(
            default="alibaba_machine_seq"
        )
    )
    target: str = fields(NonSequenceDataset).target.evolve(
        default="src.dataset.AlibabaMachineDataset"
    )


cs = ConfigStore.instance()
cs.store(
    name="dataset/base_alibaba_machine_non_seq",
    node=AlibabaMachineNonSequenceDataset,
)
cs.store(
    name="dataset/base_alibaba_machine_seq",
    node=AlibabaMachineSequenceDataset,
)

__all__ = [
    "NonSequenceDataset",
    "SequenceDataset",
    "AlibabaMachineNonSequenceDataset",
    "AlibabaMachineSequenceDataset",
]
