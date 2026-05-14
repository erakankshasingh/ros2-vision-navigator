"""Download a pre-quantized SSD MobileNet V1 (COCO, INT8 TFLite) model.

Fetches the canonical TensorFlow-hosted bundle which contains:

  - detect.tflite      : INT8-quantized SSD MobileNet V1 detector
                         input  : uint8 [1, 300, 300, 3]
                         outputs: float [1,10,4] boxes (ymin, xmin, ymax, xmax)
                                  float [1,10]   class ids (0-indexed)
                                  float [1,10]   scores (0..1)
                                  float [1]      num_detections
  - labelmap.txt       : 90 COCO class names, one per line

Skips the download if the target files already exist. Re-run with --force to
overwrite.
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


MODEL_URL = (
    'https://storage.googleapis.com/download.tensorflow.org/models/'
    'tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip'
)
MODEL_FILE = 'detect.tflite'
LABELS_FILE = 'labelmap.txt'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out',
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / 'ros2_ws' / 'src' / 'vision_navigator' / 'models',
        help='Directory to write the model and labels into.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download even if files already exist.',
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / MODEL_FILE
    labels_path = args.out / LABELS_FILE

    if not args.force and model_path.exists() and labels_path.exists():
        print(f'Already present: {model_path}, {labels_path} (use --force to overwrite)')
        return 0

    print(f'Downloading {MODEL_URL}')
    with urllib.request.urlopen(MODEL_URL) as resp:
        zip_bytes = resp.read()
    print(f'  -> {len(zip_bytes):,} bytes')

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if MODEL_FILE not in names or LABELS_FILE not in names:
            print(f'archive does not contain expected files: {names}', file=sys.stderr)
            return 1
        for name, dest in [(MODEL_FILE, model_path), (LABELS_FILE, labels_path)]:
            dest.write_bytes(zf.read(name))
            print(f'wrote {dest} ({dest.stat().st_size:,} bytes)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())