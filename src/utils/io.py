import sys
from pathlib import Path
from typing import TextIO


class Transcriber:
    def __init__(
        self, file: str | Path | TextIO | None = None, out=sys.stdout
    ):
        self.out: TextIO = out
        self.filename: str | None = None
        self.file: TextIO | None = None
        if file is not None:
            self.__setup_file__(file)

    def __setup_file__(self, file: str | Path | TextIO):
        if isinstance(file, (str, Path)):
            self.filename = str(file)
            self.file = open(file, "w")
        else:
            self.filename = file.name
            self.file = file

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def write(self, text: str):
        if self.file is not None:
            self.file.write(text)
        self.out.write(text)

    def write_line(self, text: str):
        self.write(text + "\n")

    def print(self, text: str):
        self.write(text + "\n")

    def close(self):
        if self.file is not None:
            self.file.close()

    def flush(self):
        if self.file is not None:
            self.file.flush()
        self.out.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, text: str):
        self.write(text)
        self.flush()

    def __repr__(self):
        return (
            f"Transcriber(filename={self.filename}, out={self.out})"
        )

    def __str__(self):
        return repr(self)


__all__ = ["Transcriber"]
