from avalanche.training.templates.problem_type import (
    SupervisedProblem,
)

import gorilla


@gorilla.patches(
    SupervisedProblem,
    gorilla.Settings(allow_hit=True),
)
class CustomSupervisedProblem:
    @property
    def mb_x(self: SupervisedProblem):
        return self.mbatch[0]

    @property
    def mb_y(self: SupervisedProblem):
        return self.mbatch[2]

    @property
    def mb_d(self: SupervisedProblem):
        return self.mbatch[1]

    @property
    def mb_task_id(self: SupervisedProblem):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self: SupervisedProblem):
        """Loss function."""
        return self._criterion(self.mb_output, self.mb_y)


__all__ = [
    "CustomSupervisedProblem",
]
