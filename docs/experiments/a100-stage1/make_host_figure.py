"""Visualize recorded host-test results; never fabricate GPU results."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
suite = ET.parse(root / 'host-tests/tests.xml').getroot().find('testsuite')
counts = {key: int(suite.attrib[key]) for key in ('tests', 'failures', 'errors', 'skipped')}
counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
with (root / 'host-tests/summary.json').open('x') as stream:
    json.dump(dict(counts, elapsed_s=float(suite.attrib['time']), source='tests.xml'), stream, indent=2)
fig, ax = plt.subplots(figsize=(7, 4))
labels = ['Passed', 'Failed', 'Errors', 'Skipped']
values = [counts[k] for k in ('passed', 'failures', 'errors', 'skipped')]
bars = ax.bar(labels, values, color=['#32825a', '#b94141', '#b94141', '#c59332'])
ax.bar_label(bars, padding=3)
ax.set_ylim(0, max(values) + 4)
ax.set_ylabel('Test cases')
ax.set_title('A100 stage 1: host regression only')
fig.text(.5, .02, 'Log visualization from JUnit XML; GPU tests NOT EXECUTED', ha='center', fontsize=9)
fig.tight_layout(rect=[0, .06, 1, 1])
fig.savefig(root / 'figures/host-results.png', dpi=150)
plt.close(fig)
