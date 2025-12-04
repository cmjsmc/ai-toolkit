from typing import OrderedDict, Optional
from PIL import Image

from toolkit.config_modules import LoggingConfig

class EmptyLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass
    def start(self): pass
    def log(self, *args, **kwargs): pass
    def commit(self, step: Optional[int] = None): pass
    def log_image(self, *args, **kwargs): pass
    def finish(self): pass

class WandbLogger(EmptyLogger):
    # Overridden to do nothing regardless of config
    pass

def create_logger(logging_config: LoggingConfig, all_config: OrderedDict):
    # Always return EmptyLogger to prevent external logging
    return EmptyLogger()
