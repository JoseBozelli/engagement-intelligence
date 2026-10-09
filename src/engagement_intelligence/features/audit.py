"""Point-in-time audit for feature builds (usable in tests and on real data).

The truncation audit rebuilds features for every assessment at or before a cutoff using
only events recorded at or before that cutoff, and requires exact equality with the
full-data build. A feature that uses information recorded after its assessment time
fails the audit and is named in the error.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from engagement_intelligence.core.roles import RoleMap
from engagement_intelligence.features.build import build_features
from engagement_intelligence.features.spec import FeatureSpec


def truncation_audit(spec: FeatureSpec, roles: RoleMap, events: pa.Table,
                     assessments: pa.Table, cut: datetime) -> int:
    """Raise AssertionError if any feature changes when post-cutoff records are removed.

    Returns the number of assessment rows checked.
    """
    cut_scalar = pa.scalar(cut, type=pa.timestamp("us", tz="UTC"))
    early = assessments.filter(pc.less_equal(assessments["assessed_at"], cut_scalar))
    if early.num_rows == 0:
        raise ValueError(f"no assessments at or before {cut.isoformat()}")
    truncated_events = events.filter(pc.less_equal(events["recorded_at"], cut_scalar))
    full = build_features(spec, roles, events, early)
    truncated = build_features(spec, roles, truncated_events, early)
    for name in spec.names:
        if not np.array_equal(full.column(name).to_numpy(), truncated.column(name).to_numpy()):
            raise AssertionError(f"feature {name!r} uses information recorded after {cut.isoformat()}")
    return early.num_rows