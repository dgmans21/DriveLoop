"""합성 장면으로 라벨 규칙 검증: 카메라 원점에서 +x 방향을 봄."""
import numpy as np
import pytest

from driveloop.data.autolabel import LabelRules, facing_angle, label_frame
from driveloop.data.snapshot import camera_intrinsics

W, H = 320, 180
CAM = {"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "roll": 0}
EPISODE = {
    "camera": {"width": W, "height": H, "K": camera_intrinsics(W, H, 90)},
    "traffic_lights": {"500": {"light_boxes": [
        # 정면(+y축)이 카메라(-x 방향)를 향하도록 yaw 90 → +y축이 월드 -x
        {"x": 20, "y": 0, "z": 0, "ex": 0.2, "ey": 0.2, "ez": 0.6, "pitch": 0, "yaw": 90, "roll": 0}]}},
}
ZERO = {"pitch": 0, "yaw": 0, "roll": 0}


def vehicle(vid=7, x=10.0, y=0.0):
    return {"id": vid, "transform": {"x": x, "y": y, "z": 0, **ZERO},
            "bbox": {"x": 0, "y": 0, "z": 0, "ex": 2, "ey": 1, "ez": 0.8, **ZERO},
            "distance": x, "base_type": "car"}


def frame(vehicles=(), lights=None):
    return {"camera": CAM, "vehicles": list(vehicles), "traffic_lights": lights or {},
            "ego": {"affecting_light": None}}


def seg_with(boxes):
    """boxes: [(x1,y1,x2,y2,tag,actor_id)] → (H,W,3) 인스턴스 세그."""
    seg = np.zeros((H, W, 3), np.uint8)
    for x1, y1, x2, y2, tag, aid in boxes:
        seg[y1:y2, x1:x2] = (tag, aid % 256, aid // 256)
    return seg


def test_visible_vehicle_gets_tight_box():
    seg = seg_with([(140, 80, 180, 100, 14, 7)])
    objs = label_frame(frame([vehicle()]), EPISODE, seg, LabelRules())
    assert len(objs) == 1
    assert objs[0]["cls"] == "vehicle" and objs[0]["bbox"] == [140, 80, 180, 100]


def test_fully_occluded_vehicle_is_dropped():
    seg = seg_with([(140, 80, 180, 100, 14, 8)])  # 그 자리에 다른 차(id 8)만 보임
    objs = label_frame(frame([vehicle()]), EPISODE, seg, LabelRules())
    assert [o["actor_id"] for o in objs] == [8]   # 가려진 7번은 없고, 보이는 8번만


def test_static_parked_vehicle_is_labeled_from_segmentation():
    seg = seg_with([(20, 100, 80, 130, 14, 9999)])  # 기록에 없는 차량 (맵 소품)
    objs = label_frame(frame(), EPISODE, seg, LabelRules())
    assert len(objs) == 1 and objs[0]["static"] and objs[0]["depth"] is None


def test_ego_hood_is_excluded():
    seg = seg_with([(100, 170, 220, 180, 14, 42)])  # 화면 맨 아래 중앙 = 보닛
    f = frame()
    f["ego"]["id"] = 42
    assert label_frame(f, EPISODE, seg, LabelRules()) == []


def test_find_ego_instance_from_hood_position():
    from driveloop.data.autolabel import find_ego_instance
    segs = [seg_with([(100, 170, 220, 180, 14, 42), (20, 100, 80, 130, 14, 5)]) for _ in range(3)]
    assert find_ego_instance(segs) == 42


def test_vehicle_beside_camera_kept_with_recorded_distance():
    v = vehicle(x=1.0, y=2.5)          # 3D 박스가 카메라 평면에 걸침 → 투영 불가
    v["distance"] = 2.7
    seg = seg_with([(280, 60, 320, 150, 14, 7)])
    objs = label_frame(frame([v]), EPISODE, seg, LabelRules())
    assert len(objs) == 1 and objs[0]["depth"] == 2.7 and not objs[0]["static"]


def test_large_actor_id_decoding():
    seg = seg_with([(140, 80, 180, 100, 14, 1234)])
    objs = label_frame(frame([vehicle(vid=1234)]), EPISODE, seg, LabelRules())
    assert objs and objs[0]["actor_id"] == 1234


def test_far_actor_is_dropped():
    seg = seg_with([(140, 80, 180, 100, 14, 7)])
    v = vehicle()
    v["distance"] = 120.0
    assert label_frame(frame([v]), EPISODE, seg, LabelRules()) == []


def test_traffic_light_facing_camera_is_labeled_with_state():
    seg = seg_with([(150, 70, 170, 110, 7, 500)])
    objs = label_frame(frame(lights={"500": "red"}), EPISODE, seg, LabelRules())
    assert [o["cls"] for o in objs] == ["tl_red"]
    assert objs[0]["facing_angle"] == pytest.approx(0, abs=1e-6)


def test_traffic_light_seen_from_behind_is_dropped():
    ep = {**EPISODE, "traffic_lights": {"500": {"light_boxes": [
        {**EPISODE["traffic_lights"]["500"]["light_boxes"][0], "yaw": -90}]}}}
    seg = seg_with([(150, 70, 170, 110, 7, 500)])
    assert label_frame(frame(lights={"500": "red"}), ep, seg, LabelRules()) == []


def test_traffic_light_off_is_not_labeled():
    seg = seg_with([(150, 70, 170, 110, 7, 500)])
    assert label_frame(frame(lights={"500": "off"}), EPISODE, seg, LabelRules()) == []


def test_facing_angle_side_view():
    head = {"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "roll": 0}   # 정면 = +y
    assert facing_angle(head, {"x": 0, "y": 10, "z": 0}) == pytest.approx(0)
    assert facing_angle(head, {"x": 10, "y": 0, "z": 0}) == pytest.approx(90)


def test_ignore_regions_for_visible_but_below_threshold():
    from driveloop.data.autolabel import label_frame_full
    seg = seg_with([(20, 100, 60, 106, 14, 9999),    # 납작한 차량 조각 (높이 6px) → 무시
                    (150, 70, 170, 110, 7, 500)])    # 정면 신호등
    objs, ign = label_frame_full(frame(lights={"500": "red"}), EPISODE, seg, LabelRules())
    assert [o["cls"] for o in objs] == ["tl_red"]
    assert [(i["cls"], i["reason"]) for i in ign] == [("vehicle", "too_small")]


def test_occluded_traffic_light_becomes_ignore():
    from driveloop.data.autolabel import label_frame_full
    # 헤드 투영 영역은 대략 x 158~162, y 85~95. 차량(id 3)이 y 93 위를 가리고 아래 2줄만 보임
    seg = seg_with([(150, 70, 170, 110, 7, 500), (150, 70, 170, 93, 14, 3)])
    objs, ign = label_frame_full(frame(lights={"500": "red"}), EPISODE, seg, LabelRules())
    assert not [o for o in objs if o["cls"].startswith("tl")]
    assert [(i["cls"], i["reason"]) for i in ign if i["cls"] == "traffic_light"] == [("traffic_light", "occluded")]


def test_backside_traffic_light_is_background_not_ignore():
    from driveloop.data.autolabel import label_frame_full
    ep = {**EPISODE, "traffic_lights": {"500": {"light_boxes": [
        {**EPISODE["traffic_lights"]["500"]["light_boxes"][0], "yaw": -90}]}}}
    seg = seg_with([(150, 70, 170, 110, 7, 500)])
    assert label_frame_full(frame(lights={"500": "red"}), ep, seg, LabelRules()) == ([], [])
