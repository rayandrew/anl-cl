import warnings

import torch
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.ewc import EWCPlugin
from avalanche.training.utils import zerolike_params_dict

import gorilla

from src.utils.patch import patch_filter


# @gorilla.patches(
#     EWCPlugin,
#     gorilla.Settings(allow_hit=True),
#     filter=patch_filter,
# )
class CustomEWCPlugin:
    def compute_importances(
        self: EWCPlugin,
        model,
        criterion,
        optimizer,
        dataset,
        device,
        batch_size,
        num_workers=4,
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        collate_fn = (
            dataset.collate_fn
            if hasattr(dataset, "collate_fn")
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[2], batch[-1]  # added
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))

        return importances


__all__ = ["CustomEWCPlugin"]
