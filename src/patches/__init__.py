import gorilla


def apply_patches():
    for patch in gorilla.find_patches([]):
        gorilla.apply(patch)


__all__ = ["apply_patches"]
