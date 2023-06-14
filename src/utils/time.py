from datetime import datetime


def get_current_time():
    return int(datetime.now().timestamp())


__all__ = ["get_current_time"]
