import json
from dataclasses import asdict, is_dataclass


# taken from https://stackoverflow.com/questions/51286748/make-the-python-json-encoder-support-pythons-new-dataclasses
class DataClassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


__all__ = ["DataClassJSONEncoder"]
