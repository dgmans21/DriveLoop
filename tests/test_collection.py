import json

import numpy as np
import pytest

from driveloop.config import CONFIG_DIR
from driveloop.data.collection_config import (CollectionConfig, expand_matrix, load_collection_config,
                                              weather_params)
from driveloop.data.writer import EpisodeWriter, is_complete


def make_cfg(**kw):
    base = dict(name="t", maps=["Town01", "Town03"],
                weathers={"clear": {"cloudiness": 10}, "rain": {"precipitation": 70}},
                times={"noon": {"sun_altitude_angle": 60}, "night": {"sun_altitude_angle": -40}},
                traffic={"low": 15, "high": 50})
    base.update(kw)
    return CollectionConfig(**base)


def test_v1_config_loads():
    cfg = load_collection_config(CONFIG_DIR / "collection" / "v1.yaml")
    assert len(expand_matrix(cfg)) == 16
    assert cfg.capture_every_ticks == 10


def test_matrix_is_full_product_grouped_by_map():
    specs = expand_matrix(make_cfg(repeats=2))
    assert len(specs) == 2 * 2 * 2 * 2 * 2
    maps = [s.map for s in specs]
    assert maps == sorted(maps, key=["Town01", "Town03"].index)  # 맵별로 연속
    assert len({s.episode_id for s in specs}) == len(specs)


def test_seed_stable_when_matrix_grows():
    small = {s.episode_id: s.seed for s in expand_matrix(make_cfg())}
    bigger = make_cfg(maps=["Town01", "Town02", "Town03"],
                      weathers={"clear": {}, "rain": {}, "fog": {"fog_density": 40}})
    big = {s.episode_id: s.seed for s in expand_matrix(bigger)}
    for eid, seed in small.items():
        assert big[eid] == seed


def test_time_overrides_weather():
    cfg = make_cfg(weathers={"clear": {"sun_altitude_angle": 10, "cloudiness": 5}})
    spec = [s for s in expand_matrix(cfg) if s.time == "night"][0]
    assert weather_params(cfg, spec) == {"sun_altitude_angle": -40, "cloudiness": 5}


def test_writer_atomic_finalize(tmp_path):
    ep = tmp_path / "ep1"
    rgb = np.zeros((4, 6, 3), np.uint8)
    with EpisodeWriter(ep) as w:
        for i in range(3):
            w.add(rgb, rgb, {"frame": i})
        assert not ep.exists() and not is_complete(ep)  # 완료 전에는 최종 경로가 없음
        w.finalize({"episode_id": "ep1"})
    assert is_complete(ep)
    assert not (tmp_path / "ep1.partial").exists()
    lines = (ep / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["frame"] for l in lines] == [0, 1, 2]
    assert (ep / "rgb" / "000002.jpg").exists() and (ep / "seg" / "000002.png").exists()
    assert json.loads((ep / "episode.json").read_text())["num_frames"] == 3


def test_writer_failure_leaves_no_complete_episode(tmp_path):
    ep = tmp_path / "ep2"
    with pytest.raises(RuntimeError):
        with EpisodeWriter(ep) as w:
            w.add(np.zeros((4, 6, 3), np.uint8), np.zeros((4, 6, 3), np.uint8), {})
            raise RuntimeError("simulator crashed")
    assert not is_complete(ep)
    assert (tmp_path / "ep2.partial").exists()


def test_seg_png_is_lossless(tmp_path):
    from PIL import Image
    seg = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    with EpisodeWriter(tmp_path / "ep3") as w:
        w.add(np.zeros_like(seg), seg, {})
        w.finalize({})
    assert np.array_equal(np.asarray(Image.open(tmp_path / "ep3" / "seg" / "000000.png")), seg)


def test_optional_timing_block_and_repeat_start(tmp_path):
    import yaml
    from driveloop.data.collection_config import TrafficLightTiming
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "name": "t", "maps": ["Town01"], "weathers": {"clear": {}}, "times": {"noon": {}},
        "traffic": {"low": 5}, "repeat_start": 1, "repeats": 2,
        "traffic_light_timing": {"green": 6, "yellow": 6, "red": 2}, "yellow_capture_interval": 0.1}))
    cfg = load_collection_config(p)
    assert isinstance(cfg.traffic_light_timing, TrafficLightTiming) and cfg.traffic_light_timing.yellow == 6
    assert cfg.yellow_capture_ticks == 2
    assert [s.repeat for s in expand_matrix(cfg)] == [1, 2]


def test_new_repeat_gets_new_seed_old_ids_unchanged():
    v1 = {s.episode_id: s.seed for s in expand_matrix(make_cfg())}
    v2 = expand_matrix(make_cfg(repeat_start=1))
    assert all(s.episode_id.endswith("_r1") and s.episode_id not in v1 for s in v2)


def test_episode_length_is_time_based():
    cfg = make_cfg()
    assert cfg.episode_ticks == cfg.frames_per_episode * cfg.capture_every_ticks
