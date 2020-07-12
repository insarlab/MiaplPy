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

sg_logger = logging.getLogger('shapely.geos')
sg_logger.setLevel(logging.WARNING)

as_logger = logging.getLogger('asyncio')
as_logger.setLevel(logging.WARNING)

# expose the following modules
__all__ = [
    'crop_images',
    'generate_interferograms',
    'phase_inversion',
    'load_int',
    'version',
]

root_module = Path(__file__).parent.parent.name   #minopy
for module in __all__:
    importlib.import_module(root_module + '.' + module)

