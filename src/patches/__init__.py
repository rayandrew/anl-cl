import gorilla

import src.patches.skmultiflow as patches_skmultiflow


def apply_patches():
    for patch in gorilla.find_patches([patches_skmultiflow]):
        print("Applying patch", patch)
        gorilla.apply(patch)


__all__ = ["apply_patches"]
