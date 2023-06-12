import torch


def get_device_from_config(cfg: dict):
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


__all__ = ["get_device_from_config"]
