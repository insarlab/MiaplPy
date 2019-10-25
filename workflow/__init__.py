## Dynamic import for modules used in routine workflows
## Recommended usage:
##     import minopy
##     import minopy.workflow


from pathlib import Path
import importlib


# expose the following modules
__all__ = [
    'create_patch',
    'crop_images',
    'generate_ifgram_sq',
    'patch_inversion',
    'version',
]

root_module = Path(__file__).parent.parent.name   #minopy
for module in __all__:
    importlib.import_module(root_module + '.' + module)

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
