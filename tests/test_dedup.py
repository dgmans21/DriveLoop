from driveloop.data.dedup import DedupRules, dedup_episode

H0 = "0" * 16
H_NEAR = "0000000000000003"   # 해밍 2
H_FAR = "00000000000000ff"    # 해밍 8


def fr(i, t=None, x=0.0, h=H0, red=1, veh=0):
    return {"index": i, "t": i * 0.5 if t is None else t, "ego_x": x, "ego_y": 0.0, "dhash": h,
            "n_vehicle": veh, "n_tl_red": red, "n_tl_yellow": 0, "n_tl_green": 0}


def keeps(frames, **kw):
    return [r["index"] for r in dedup_episode(frames, DedupRules(**kw)) if r["keep"]]


def test_stopped_identical_frames_collapse_to_first():
    assert keeps([fr(i) for i in range(5)]) == [0]


def test_light_change_is_kept():
    frames = [fr(0), fr(1), fr(2, red=0), fr(3, red=0)]
    assert keeps(frames) == [0, 2]


def test_moving_frames_are_kept_even_if_image_similar():
    frames = [fr(i, x=i * 4.0) for i in range(4)]   # 0.5초마다 4m 이동, 해시는 동일
    assert keeps(frames) == [0, 1, 2, 3]


def test_compares_to_representative_not_previous():
    # 조금씩 변해서 직전과는 비슷하지만 대표와는 멀어지는 경우
    frames = [fr(0, h=H0), fr(1, h=H_NEAR), fr(2, h=H_FAR)]
    assert keeps(frames) == [0, 2]


def test_max_gap_keeps_periodic_sample_while_waiting():
    frames = [fr(i) for i in range(50)]             # 25초 정지
    assert keeps(frames, max_gap_s=10.0) == [0, 20, 40]


def test_dup_of_points_to_representative():
    res = dedup_episode([fr(0), fr(1), fr(2)], DedupRules())
    assert [r["dup_of"] for r in res] == [None, 0, 0]
