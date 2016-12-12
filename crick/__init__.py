from __future__ import absolute_import

from .tdigest import TDigest
from .stream_summary import StreamSummary
from .stats import SummaryStats

from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
