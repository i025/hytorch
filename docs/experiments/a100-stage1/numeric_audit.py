"""Optional pytest plugin: retain comparison metrics without changing assertions."""
import json
import os
from pathlib import Path
import numpy as np
import pytest


def pytest_sessionstart(session):
    Path(os.environ['HYTORCH_NUMERIC_AUDIT']).open('x').close()


@pytest.fixture(autouse=True)
def record_numeric_comparisons(request, monkeypatch):
    path = Path(os.environ['HYTORCH_NUMERIC_AUDIT'])
    for name in ('assert_allclose', 'assert_array_equal'):
        original = getattr(np.testing, name)

        def audited(actual, desired, *args, _name=name, _original=original, **kwargs):
            a, b = np.asarray(actual, dtype=np.float64), np.asarray(desired, dtype=np.float64)
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            row = dict(test=request.node.nodeid, assertion=_name, shape=list(a.shape),
                       elements=int(a.size), finite=finite)
            if finite:
                difference = a-b
                row.update(max_abs_error=float(np.max(np.abs(difference))),
                           relative_l2_error=float(np.linalg.norm(difference.ravel()) /
                                                   max(np.linalg.norm(b.ravel()), 1e-30)))
            if _name == 'assert_allclose':
                rtol = args[0] if args else kwargs.get('rtol', 1e-7)
                atol = args[1] if len(args) > 1 else kwargs.get('atol', 0)
                row.update(rtol=rtol, atol=atol)
            with path.open('a') as log:
                log.write(json.dumps(row, allow_nan=False) + '\n')
            return _original(actual, desired, *args, **kwargs)

        monkeypatch.setattr(np.testing, name, audited)
