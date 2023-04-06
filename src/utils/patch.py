import types
from typing import Any


def patch_filter(name: str, obj: Any):
    """Attribute filter.
    It filters out module attributes, and also methods starting with an
    underscore ``_``.
    This is used as the default filter for the :func:`create_patches` function
    and the :func:`patches` decorator.
    Parameters
    ----------
    name : str
        Attribute name.
    obj : object
        Attribute value.
    Returns
    -------
    bool
        Whether the attribute should be returned.
    """
    cond = (
        not (
            isinstance(obj, types.ModuleType) or name.startswith("_")
        )
        or name == "__init__"
    )

    return cond
