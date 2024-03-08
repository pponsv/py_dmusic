from .src import dmusic_functions as dm
from .src import dmusic_io
from .src.dmusic_class import DMusic

try:
    from .src.dmusic_torch import DMusic_gpu
except:
    print("PyTorch not found, DMusic_gpu not available")
