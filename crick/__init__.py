from __future__ import absolute_import

from ._version import get_versions
from .space_saving import SpaceSaving
from .stats import SummaryStats
from .tdigest import TDigest

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
