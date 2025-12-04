import sys
import os
from toolkit.accelerator import get_accelerator

# PREVIOUS: Printed to stdout/log file
# NEW: Silenced completely for security/privacy
def print_acc(*args, **kwargs):
    pass

class Logger:
    def __init__(self, filename):
        pass

    def write(self, message):
        pass

    def flush(self):
        pass

def setup_log_to_file(filename):
    pass
