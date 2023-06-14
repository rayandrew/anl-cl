import torch


def get_device_from_config_dict(cfg: dict):
    device = "cpu"
    cuda_config = cfg.get("cuda", {})
    if (
        "enable" in cuda_config
        and "device" in cuda_config
        and cuda_config["enable"]
        and cuda_config["device"] >= 0
        and torch.cuda.is_available()
    ):
        device = f"cuda:{cuda_config['device']}"

    return torch.device(device)


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# https://github.com/RaptorMai/online-continual-learning/blob/6175ca034e56435acd82b8f17ff59f920f0bc45e/utils/utils.py#L4
def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what


__all__ = ["get_device_from_config_dict", "get_device", "maybe_cuda"]
