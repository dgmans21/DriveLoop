"""검출·판단 오버레이 (numpy + OpenCV). 화면 표시와 MP4 녹화가 같은 프레임을 쓴다.

모든 함수는 RGB 이미지를 받아 RGB를 돌려준다.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# RGB. 차량=파랑, 신호 색=같은 색 (리포트·검출 영상과 동일)
COLORS = {"vehicle": (42, 120, 214), "tl_red": (227, 73, 72), "tl_yellow": (237, 161, 0), "tl_green": (0, 170, 0)}
STATE_RGB = {"RED": (227, 73, 72), "YELLOW": (237, 161, 0), "GREEN": (0, 170, 0), "UNKNOWN": (170, 170, 170)}
SHORT = {"vehicle": "car", "tl_red": "red", "tl_yellow": "yellow", "tl_green": "green"}


def _text(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    # 테두리는 같은 굵기로 주변 8방향에 그린다 — Hershey 글꼴은 굵기가 커지면 글자 간격도 넓어져서
    # 굵은 테두리 문자열이 본문보다 길어지고 끝에 잔상이 남는다 ("TRUTH H")
    x, y = org
    for dx in (-2, 0, 2):
        for dy in (-2, 0, 2):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _box(b):
    return (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))


def draw_detections(img: np.ndarray, detections, expected=(), chosen=None) -> np.ndarray:
    """detections: .cls/.box/.conf 를 가진 객체들. expected: 지도로 예상한 내 신호 위치. chosen: 고른 검출."""
    out = img.copy()
    for e in expected:
        p1, p2 = _box(e)
        cv2.rectangle(out, (p1[0] - 3, p1[1] - 3), (p2[0] + 3, p2[1] + 3), (255, 255, 255), 1, cv2.LINE_AA)
    for d in detections:
        p1, p2 = _box(d.box)
        cv2.rectangle(out, p1, p2, COLORS.get(d.cls, (255, 255, 255)), 1, cv2.LINE_AA)
    if chosen is not None:
        p1, p2 = _box(chosen.box)
        col = COLORS.get(chosen.cls, (255, 255, 255))
        cv2.rectangle(out, (p1[0] - 2, p1[1] - 2), (p2[0] + 2, p2[1] + 2), col, 3, cv2.LINE_AA)
        _text(out, f"MY LIGHT {SHORT.get(chosen.cls, chosen.cls)} {chosen.conf:.2f}", (p1[0], max(16, p1[1] - 8)),
              0.55, col, 2)
    return out


def draw_panel(img: np.ndarray, lines: Sequence[tuple[str, tuple[int, int, int]]], x: int = 10, y: int = 10,
               width: int = 430) -> np.ndarray:
    out = img.copy()
    h = 26 * len(lines) + 14
    roi = out[y:y + h, x:x + width]
    out[y:y + h, x:x + width] = (roi * 0.4).astype(np.uint8)
    for i, (t, col) in enumerate(lines):
        _text(out, t, (x + 10, y + 26 + 26 * i), 0.6, col, 1)
    return out


def paste_inset(img: np.ndarray, inset: np.ndarray, scale: float = 0.3, margin: int = 10) -> np.ndarray:
    """오른쪽 아래에 작은 화면 (예: 3인칭 시점)."""
    out = img.copy()
    h, w = img.shape[:2]
    iw, ih = int(w * scale), int(inset.shape[0] * (w * scale) / inset.shape[1])
    small = cv2.resize(inset, (iw, ih), interpolation=cv2.INTER_AREA)
    y0, x0 = h - ih - margin, w - iw - margin
    out[y0:y0 + ih, x0:x0 + iw] = small
    cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), (255, 255, 255), 1)
    return out


def to_web_mp4(src, dst, crf: int = 23) -> bool:
    """OpenCV 녹화(mp4v)는 브라우저에서 재생되지 않는다 → ffmpeg로 H.264 변환. ffmpeg가 없으면 False."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", str(crf),
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    return True


class Mp4Recorder:
    def __init__(self, path, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self._w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        self._w.write(np.ascontiguousarray(rgb[:, :, ::-1]))
        self.frames += 1

    def close(self) -> None:
        self._w.release()
