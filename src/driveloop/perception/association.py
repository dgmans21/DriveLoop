"""'내 신호등' 고르기 + 시간 축 안정화 (CARLA·YOLO 비의존).

지도(HD map 역할)는 내 차선을 통제하는 신호등의 3D 위치를 알고, 모델은 색을 안다.
지도의 신호등 헤드를 화면에 투영한 '예상 위치' 근처의 검출만 내 신호로 인정한다.
→ 다른 차선 신호, 노면 반사광 같은 엉뚱한 위치의 오검출이 자연스럽게 걸러진다.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from driveloop.data.autolabel import facing_angle
from driveloop.data.geometry import box_corners, box_to_2d
from driveloop.eval.detection import iou_matrix

Box = tuple[float, float, float, float]
TL_STATES = {"tl_red": "RED", "tl_yellow": "YELLOW", "tl_green": "GREEN"}


@dataclass(frozen=True)
class Detection:
    cls: str
    box: Box
    conf: float


@dataclass(frozen=True)
class Association:
    state: str          # RED / YELLOW / GREEN
    conf: float
    det: Detection
    expected: Box       # 이 검출과 짝지어진 예상 위치


def expected_head_boxes(heads: list[dict], camera_tf: dict, K: np.ndarray, width: int, height: int,
                        max_facing_angle: float = 70.0) -> list[Box]:
    """내 신호등 헤드(월드 3D 박스)들 중 카메라를 향하고 화면 안에 있는 것의 2D 예상 박스."""
    out = []
    for h in heads:
        if facing_angle(h, camera_tf) > max_facing_angle:
            continue
        proj = box_to_2d(box_corners(h), camera_tf, K, width, height)
        if proj is not None:
            out.append(proj[0])
    return out


def associate(expected: list[Box], detections: list[Detection], min_iou: float = 0.1,
              max_center_dist: float = 1.5) -> Association | None:
    """예상 박스와 가장 잘 맞는 신호등 검출.

    짝 조건: IoU ≥ min_iou, 또는 중심 거리가 예상 박스 높이 × max_center_dist 이내
    (먼 신호등은 박스가 작아 투영 오차 몇 px에도 IoU가 0이 되므로 거리 기준을 같이 쓴다).
    점수 = IoU + 신뢰도. 여러 헤드가 맞으면 점수가 가장 높은 것.
    """
    tl = [d for d in detections if d.cls in TL_STATES]
    if not expected or not tl:
        return None
    E = np.array(expected, dtype=float)
    D = np.array([d.box for d in tl], dtype=float)
    ious = iou_matrix(E, D)
    ec = (E[:, :2] + E[:, 2:]) / 2
    dc = (D[:, :2] + D[:, 2:]) / 2
    dist = np.linalg.norm(ec[:, None, :] - dc[None, :, :], axis=2)
    eh = np.maximum(E[:, 3] - E[:, 1], 1.0)[:, None]
    ok = (ious >= min_iou) | (dist <= eh * max_center_dist)
    best = None
    for i, j in zip(*np.nonzero(ok)):
        score = ious[i, j] + tl[j].conf
        if best is None or score > best[0]:
            best = (score, i, j)
    if best is None:
        return None
    _, i, j = best
    return Association(TL_STATES[tl[j].cls], tl[j].conf, tl[j], tuple(expected[i]))


class TemporalVoter:
    """최근 window 프레임의 관측을 다수결로 안정화.

    - 관측 없음(None)은 투표에 넣지 않는다 (한두 프레임 놓쳐도 직전 판단 유지)
    - 유효 관측이 min_votes 미만이면 UNKNOWN
    - RED 우선: RED가 red_votes 이상이면 다수결과 무관하게 RED (놓치면 신호 위반이므로 보수적으로)
    - YELLOW 우선: RED가 아니고 YELLOW가 yellow_votes 이상이면 GREEN보다 YELLOW
      → 안전 순서 RED > YELLOW > GREEN. Town05 주행에서 노란불을 3프레임 GREEN으로 오인해
        0.15초 정지를 풀었던 사례 (2026-10-07). 실제 신호는 노랑 다음이 항상 빨강이라 손해가 없다
    """

    def __init__(self, window: int = 5, min_votes: int = 2, red_votes: int = 2, yellow_votes: int = 2) -> None:
        self._hist: deque[str | None] = deque(maxlen=window)
        self.min_votes = min_votes
        self.red_votes = red_votes
        self.yellow_votes = yellow_votes

    def reset(self) -> None:
        self._hist.clear()

    def update(self, state: str | None) -> str:
        self._hist.append(state)
        votes = Counter(s for s in self._hist if s is not None)
        if sum(votes.values()) < self.min_votes:
            return "UNKNOWN"
        if votes["RED"] >= self.red_votes:
            return "RED"
        if votes["YELLOW"] >= self.yellow_votes:
            return "YELLOW"
        return votes.most_common(1)[0][0]
