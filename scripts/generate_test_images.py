"""Generate synthetic test images for the camera_node.

Produces four small (320x240) BGR PNGs covering different visual patterns
so the pipeline can be smoke-tested before real photos are available:

  - solid_red.png    : flat solid color (uniform pixels)
  - gradient.png     : horizontal gradient (smooth intensity ramp)
  - stripes.png      : alternating vertical stripes (high-frequency content)
  - circle.png       : centred filled circle on dark background (a "blob")

Idempotent: re-running overwrites existing files. Safe to re-run any time.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


WIDTH = 320
HEIGHT = 240


def solid_red() -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 200)  # BGR
    return img


def gradient() -> np.ndarray:
    ramp = np.linspace(0, 255, WIDTH, dtype=np.uint8)
    img = np.tile(ramp, (HEIGHT, 1))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def stripes() -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    stripe_width = 20
    for x in range(0, WIDTH, stripe_width * 2):
        img[:, x:x + stripe_width] = (255, 255, 255)
    return img


def circle() -> np.ndarray:
    img = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
    cv2.circle(img, (WIDTH // 2, HEIGHT // 2), 60, (0, 200, 200), thickness=-1)
    return img


IMAGES = {
    'solid_red.png': solid_red,
    'gradient.png': gradient,
    'stripes.png': stripes,
    'circle.png': circle,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out',
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / 'ros2_ws' / 'src' / 'vision_navigator' / 'test_images',
        help='Directory to write images into.',
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, fn in IMAGES.items():
        path = args.out / name
        cv2.imwrite(str(path), fn())
        print(f'wrote {path}')


if __name__ == '__main__':
    main()