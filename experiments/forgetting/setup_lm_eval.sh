#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base_python="$repo/.venv/bin/python"
deps="$repo/.forgetting-deps"

[[ -x "$base_python" ]] || { echo "Missing training environment: $base_python" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }

# Keep evaluation-only packages outside the live training environment. The
# target directory takes precedence through PYTHONPATH and avoids downgrading
# packages used by ongoing training jobs.
mkdir -p "$deps"
uv pip install --target "$deps" --python "$base_python" --no-deps \
  'lm-eval==0.4.13' 'absl-py==2.5.0' 'chardet==6.0.0.post1' \
  'colorama==0.4.6' 'dataproperty==1.1.1' 'defusedxml==0.7.1' \
  'evaluate==0.4.6' 'joblib==1.6.0' 'more-itertools==11.1.0' \
  'narwhals==2.26.0' 'nltk==3.10.3' 'pathvalidate==3.3.1' \
  'portalocker==4.3.0' 'pytablewriter==1.2.1' 'pytz==2026.3.post1' \
  'rouge-score==0.1.2' 'sacrebleu==2.6.0' 'scikit-learn==1.9.0' \
  'sqlitedict==1.7.0' 'tabledata==1.3.5' 'tcolorpy==0.1.7' \
  'threadpoolctl==3.6.0' 'typepy==1.3.5' 'word2number==1.1'

PYTHONPATH="$deps${PYTHONPATH:+:$PYTHONPATH}" "$base_python" -m lm_eval --help >/dev/null
echo "lm-eval environment ready; prefix commands with: PYTHONPATH=$deps"
