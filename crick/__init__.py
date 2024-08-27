from ._version import get_versions
from .numpy_version import numpy_version
from .space_saving import SpaceSaving
from .stats import SummaryStats
from .tdigest import TDigest

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

from . import _version

__version__ = _version.get_versions()["version"]

__numpy_version__ = numpy_version()
