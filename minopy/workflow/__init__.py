## Dynamic import for modules used in routine workflows
## Recommended usage:
##     import minopy
##     import minopy.workflow


from pathlib import Path
import logging
import warnings
import importlib


warnings.filterwarnings("ignore")

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

fi_logger = logging.getLogger('fiona._env')
fi_logger.setLevel(logging.DEBUG)

sg_logger = logging.getLogger('shapely.geos')
sg_logger.setLevel(logging.WARNING)

as_logger = logging.getLogger('asyncio')
as_logger.setLevel(logging.WARNING)

# expose the following modules
__all__ = [
    'load_slc',
    'generate_interferograms',
    'generate_unwrap_mask',
    'phase_inversion',
    'load_ifgram',
    'unwrap_minopy',
    'phase_to_range',
    'find_short_baselines',
    'version',
]

root_module = Path(__file__).parent.parent.name   #minopy
for module in __all__:
    importlib.import_module(root_module + '.' + module)

