import numpy as np

from driveloop.perception.association import Detection
from driveloop.viz.overlay import Mp4Recorder, draw_detections, draw_panel, paste_inset


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
