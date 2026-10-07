import shutil

import numpy as np
import pytest

from driveloop.perception.association import Detection
from driveloop.viz.overlay import Mp4Recorder, draw_detections, draw_panel, paste_inset, to_web_mp4


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 없음")
def test_to_web_mp4_converts_for_browser(tmp_path):
    rec = Mp4Recorder(tmp_path / "raw.mp4", 20, (64, 48))
    for i in range(10):
        rec.write(np.full((48, 64, 3), i * 20, np.uint8))
    rec.close()
    assert to_web_mp4(rec.path, tmp_path / "web.mp4")
    assert (tmp_path / "web.mp4").stat().st_size > 0


def test_overlay_functions_keep_shape_and_do_not_modify_input():
    img = np.zeros((180, 320, 3), np.uint8)
    det = Detection("tl_red", (150, 80, 160, 100), 0.9)
    out = draw_detections(img, [det, Detection("vehicle", (10, 10, 50, 40), 0.8)], [(149, 79, 161, 101)], det)
    out = draw_panel(out, [("Light : RED", (255, 0, 0))])
    out = paste_inset(out, np.full((90, 160, 3), 200, np.uint8), 0.3)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert img.sum() == 0 and out.sum() > 0


def test_recorder_writes_file(tmp_path):
    rec = Mp4Recorder(tmp_path / "a.mp4", 20, (320, 180))
    for _ in range(3):
        rec.write(np.zeros((180, 320, 3), np.uint8))
    rec.close()
    assert rec.frames == 3 and (tmp_path / "a.mp4").stat().st_size > 0
