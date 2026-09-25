"""RLE encode/decode (app/rle.py) is easy to get subtly wrong: row-major vs
column-major, 0- vs 1-indexed starts, off-by-one run lengths. A bug here
would silently corrupt every mask without crashing anything, so it's tested
against a real row from the actual competition data, not just synthetic
examples, since that's the only way to catch a convention mismatch.
"""
import numpy as np
import pandas as pd
import os

from app.rle import rle_decode, rle_encode, IMG_HEIGHT, IMG_WIDTH


def test_decode_of_empty_string_is_an_all_zero_mask():
    mask=rle_decode("", shape=(4,4))
    assert mask.shape==(4,4)
    assert mask.sum()==0


def test_decode_of_nan_is_an_all_zero_mask():
    mask=rle_decode(float("nan"), shape=(4,4))
    assert mask.sum()==0


def test_simple_known_encoding_decodes_to_the_expected_pixels():
    # a 4x4 image, column-major flattening: pixel index 0 = (row0,col0), index 4 = (row0,col1), etc.
    # "1 2" means pixels 0-1 (1-indexed start=1, length=2) are set -> (row0,col0) and (row1,col0)
    mask=rle_decode("1 2", shape=(4,4))
    expected=np.zeros((4,4), dtype=np.uint8)
    expected[0,0]=1
    expected[1,0]=1
    np.testing.assert_array_equal(mask, expected)


def test_encode_decode_round_trip_on_a_synthetic_mask():
    mask=np.zeros((20,30), dtype=np.uint8)
    mask[5:10, 3:8]=1
    mask[15:18, 20:25]=1

    encoded=rle_encode(mask)
    decoded=rle_decode(encoded, shape=(20,30))

    np.testing.assert_array_equal(mask, decoded)


def test_encode_decode_round_trip_on_a_real_competition_row():
    csv_path=os.path.join(os.path.dirname(__file__), "..", "severstal-steel-defect-detection", "train.csv")
    df=pd.read_csv(csv_path, nrows=50)
    row=df.iloc[0]

    decoded=rle_decode(row["EncodedPixels"], shape=(IMG_HEIGHT, IMG_WIDTH))
    assert decoded.shape==(IMG_HEIGHT, IMG_WIDTH)
    assert decoded.sum()>0 # this row has a real, non-empty defect mask

    re_encoded=rle_encode(decoded)
    assert re_encoded==row["EncodedPixels"].strip()
