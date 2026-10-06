import numpy as np
from PIL import Image

from driveloop.data.metadata import QCRules, dhash, hamming, lamp_color_status, luma_stats, object_row


def solid(rgb, size=(20, 40)):
    return Image.fromarray(np.full((size[1], size[0], 3), rgb, np.uint8))


def lamp(color):
    """어두운 헤드 위에 켜진 불빛 하나."""
    a = np.full((40, 20, 3), 30, np.uint8)
    a[5:12, 6:14] = color
    return Image.fromarray(a)


def test_lamp_color_match_and_mismatch():
    assert lamp_color_status(lamp((255, 30, 30)), "tl_red") == "match"
    assert lamp_color_status(lamp((40, 255, 120)), "tl_green") == "match"
    assert lamp_color_status(lamp((255, 220, 0)), "tl_yellow") == "match"
    assert lamp_color_status(lamp((40, 255, 120)), "tl_red") == "mismatch"


def test_washed_out_lamp_in_rain_haze_still_matches():
    a = np.full((40, 20, 3), 150, np.uint8)      # 뿌연 회색 헤드
    a[5:12, 6:14] = (225, 150, 165)               # 채도 낮은 분홍빛 빨간불
    assert lamp_color_status(Image.fromarray(a), "tl_red") == "match"


def test_lamp_saturated_and_dark():
    assert lamp_color_status(lamp((250, 250, 250)), "tl_red") == "saturated"
    assert lamp_color_status(solid((30, 30, 30)), "tl_red") == "dark"


def test_dhash_similar_vs_different():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
    noisy = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
    other = rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
    h0, h1, h2 = (dhash(Image.fromarray(a)) for a in (base, noisy, other))
    assert hamming(h0, h1) < 8 < hamming(h0, h2)


def test_luma_stats_blank():
    mean, std = luma_stats(solid((0, 0, 0), (64, 36)))
    assert mean == 0 and std == 0


def test_object_row_flags():
    img = lamp((255, 30, 30)).resize((100, 100))
    veh = {"cls": "vehicle", "bbox": [0, 0, 95, 95], "actor_id": 1, "depth": 2, "visible_px": 9000,
           "visible_ratio": 0.9}
    assert "vehicle_box_too_large" in object_row(veh, img, 100, 100, QCRules())["qc_flags"]
    flat_tl = {"cls": "tl_red", "bbox": [10, 10, 40, 15], "actor_id": 2, "depth": 20, "visible_px": 100,
               "visible_ratio": 0.5, "facing_angle": 5, "affects_ego": False}
    assert "tl_aspect_odd" in object_row(flat_tl, img, 100, 100, QCRules())["qc_flags"]
