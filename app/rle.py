"""Run-length encoding for the Severstal Steel Defect Detection dataset.

The mask format is pairs of (start_pixel, run_length) in column-major order
over a flattened 256x1600 image. This matches the competition's own data
description, not a generic RLE convention.
"""
import numpy as np

IMG_HEIGHT=256
IMG_WIDTH=1600


def rle_decode(rle_string: str, shape: tuple[int, int]=(IMG_HEIGHT, IMG_WIDTH)) -> np.ndarray:
    """Decode a Severstal-format RLE string into a binary (H, W) mask."""
    if not rle_string or (isinstance(rle_string, float)): # pandas gives NaN for empty cells
        return np.zeros(shape, dtype=np.uint8)

    parts=rle_string.split()
    starts=np.array(parts[0::2], dtype=int)-1 # RLE is 1-indexed
    lengths=np.array(parts[1::2], dtype=int)

    mask=np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for start,length in zip(starts,lengths):
        mask[start:start+length]=1

    return mask.reshape(shape, order="F") # column-major


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary (H, W) mask into a Severstal-format RLE string.

    Inverse of rle_decode. Round-tripping a mask through encode then decode
    should give back the same mask, which is what tests/test_rle.py checks.
    """
    pixels=mask.flatten(order="F")
    pixels=np.concatenate([[0], pixels, [0]]) # pad so a run touching the edge is still caught
    runs=np.where(pixels[1:]!=pixels[:-1])[0]+1
    runs[1::2]-=runs[0::2] # turn (start_of_run, start_of_next_run) pairs into (start, length)
    return " ".join(str(x) for x in runs)
