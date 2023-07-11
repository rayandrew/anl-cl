import json
from dataclasses import asdict, is_dataclass
from enum import Enum, EnumMeta


# taken from https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses
class DataClassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass


class StrEnum(str, BaseEnum):
    def __str__(self):
        return self.value


__all__ = ["DataClassJSONEncoder", "StrEnum"]
