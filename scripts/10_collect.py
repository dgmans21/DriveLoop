"""2-1: 수집 매트릭스대로 원본 데이터를 배치 수집한다.

    python scripts/10_collect.py --dry-run                 # 조합/예상 규모만 출력 (CARLA 불필요)
    python scripts/10_collect.py --limit 1                 # 파일럿: 1개 에피소드만
    python scripts/10_collect.py                           # 전체 (완료된 에피소드는 건너뜀)
    python scripts/10_collect.py --only Town01_rain        # ID에 문자열이 포함된 에피소드만

자차와 NPC 모두 Traffic Manager 자동주행 (자차는 신호 준수).
"""
import argparse
import itertools
import queue
import random
import time
from contextlib import nullcontext

import numpy as np

import carla

from driveloop.config import CONFIG_DIR, load_sim_config
from driveloop.data.capture_policy import CaptureScheduler, yellow_in_view
from driveloop.data.collection_config import (CollectionConfig, EpisodeSpec, expand_matrix,
                                              load_collection_config, weather_params)
from driveloop.data.snapshot import (camera_intrinsics, frame_snapshot, traffic_light_catalog,
                                     transform_to_dict)
from driveloop.data.writer import EpisodeWriter, is_complete
from driveloop.sim.client import ActorPool, connect, spawn_ego, synchronous_mode
from driveloop.sim.sensors import get_frame, image_to_rgb
from driveloop.sim.traffic import spawn_npc_vehicles, spawn_npc_walkers, traffic_light_timing, traffic_manager
from driveloop.sim.weather import apply_weather, weather_to_dict

TM_PORT = 8000
# 디스크 추정치 (1280x720 기준 jpg + 세그 png + 메타)
EST_MB_PER_FRAME = 0.35


def attach_camera(world, ego, blueprint_id, cam_cfg):
    bp = world.get_blueprint_library().find(blueprint_id)
    bp.set_attribute("image_size_x", str(cam_cfg.width))
    bp.set_attribute("image_size_y", str(cam_cfg.height))
    bp.set_attribute("fov", str(cam_cfg.fov))
    tf = carla.Transform(carla.Location(x=cam_cfg.x, z=cam_cfg.z))
    return world.spawn_actor(bp, tf, attach_to=ego)


def run_episode(client, world, tm, cfg: CollectionConfig, spec: EpisodeSpec, sim_cfg) -> dict:
    rng = random.Random(spec.seed)
    tm.set_random_device_seed(spec.seed)
    params = weather_params(cfg, spec)
    apply_weather(world, params)

    timing = cfg.traffic_light_timing
    timing_ctx = (traffic_light_timing(world, timing.green, timing.yellow, timing.red)
                  if timing else nullcontext())
    with timing_ctx, ActorPool() as pool:
        ego, spawn_idx = spawn_ego(world, sim_cfg, rng)
        pool.add(ego)
        npc_ids = spawn_npc_vehicles(client, world, tm, cfg.traffic[spec.traffic], rng)
        for actor in world.get_actors(npc_ids):
            pool.add(actor)
        walker_ids, ctrl_ids = spawn_npc_walkers(client, world, cfg.walker_count(spec.traffic), rng,
                                                 cfg.walker_cross_factor, cfg.walker_running, spec.seed)
        for actor in list(world.get_actors(walker_ids)) + list(world.get_actors(ctrl_ids)):
            pool.add(actor)                      # 컨트롤러가 나중에 추가 → 먼저 정리(stop)된다
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego, 0)
        tm.update_vehicle_lights(ego, True)

        rgb_cam = pool.add(attach_camera(world, ego, "sensor.camera.rgb", cfg.camera))
        seg_cam = pool.add(attach_camera(world, ego, "sensor.camera.instance_segmentation", cfg.camera))
        rgb_q, seg_q = queue.Queue(), queue.Queue()
        rgb_cam.listen(rgb_q.put)
        seg_cam.listen(seg_q.put)

        traffic_lights = list(world.get_actors().filter("traffic.traffic_light"))
        catalog = traffic_light_catalog(world)
        K = np.array(camera_intrinsics(cfg.camera.width, cfg.camera.height, cfg.camera.fov))
        scheduler = CaptureScheduler(cfg.capture_every_ticks, cfg.yellow_capture_ticks)
        episode_dir = cfg.output_dir / spec.episode_id
        start_wall = time.time()
        reasons = {"regular": 0, "yellow": 0}

        with EpisodeWriter(episode_dir) as writer:
            t0 = None
            for tick in range(cfg.warmup_ticks + cfg.episode_ticks):
                frame = world.tick()
                rgb_img = get_frame(rgb_q, frame)   # 매 tick 꺼내야 큐가 쌓이지 않는다
                seg_img = get_frame(seg_q, frame)
                if tick < cfg.warmup_ticks:
                    continue
                k = tick - cfg.warmup_ticks
                yellow = False
                if scheduler.wants_fast_check(k):   # 비용이 드는 판정은 저장 가능한 시점에만
                    ego_loc = ego.get_location()
                    states = {str(tl.id): "yellow" for tl in traffic_lights
                              if tl.get_state() == carla.TrafficLightState.Yellow
                              and tl.get_location().distance(ego_loc) <= cfg.label_radius}
                    yellow = bool(states) and yellow_in_view(
                        states, catalog, transform_to_dict(rgb_cam.get_transform()), K,
                        cfg.camera.width, cfg.camera.height, cfg.yellow_max_facing_angle, cfg.label_radius)
                reason = scheduler.decide(k, yellow)
                if reason is None:
                    continue
                reasons[reason] += 1
                sim_t = world.get_snapshot().timestamp.elapsed_seconds
                t0 = sim_t if t0 is None else t0
                meta = {"frame": frame, "t": round(sim_t - t0, 3), "capture_reason": reason,
                        **frame_snapshot(world, ego, rgb_cam, traffic_lights, cfg.label_radius)}
                writer.add(image_to_rgb(rgb_img), image_to_rgb(seg_img), meta)

            wall = time.time() - start_wall
            writer.finalize({
                "episode_id": spec.episode_id,
                "dataset": cfg.name,
                "tags": spec.tags(),
                "seed": spec.seed,
                "spawn_point": spawn_idx,
                "npc_vehicles": len(npc_ids),
                "npc_walkers": len(walker_ids),
                "walker_cross_factor": cfg.walker_cross_factor if walker_ids else None,
                "weather": weather_to_dict(world.get_weather()),
                "weather_config": params,
                "camera": {"width": cfg.camera.width, "height": cfg.camera.height, "fov": cfg.camera.fov,
                           "mount": {"x": cfg.camera.x, "z": cfg.camera.z},
                           "K": camera_intrinsics(cfg.camera.width, cfg.camera.height, cfg.camera.fov)},
                "fixed_delta_seconds": cfg.fixed_delta_seconds,
                "capture_interval": cfg.capture_interval,
                "yellow_capture_interval": cfg.yellow_capture_interval,
                "traffic_light_timing": None if timing is None else
                    {"green": timing.green, "yellow": timing.yellow, "red": timing.red},
                "capture_counts": reasons,
                "label_radius": cfg.label_radius,
                "traffic_lights": catalog,
                "carla_version": client.get_server_version(),
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "wall_seconds": round(wall, 1),
            })
    return {"frames": writer.count, "wall": wall, "npc": len(npc_ids), "walkers": len(walker_ids),
            "yellow": reasons["yellow"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_DIR / "collection" / "v1.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 에피소드 수")
    parser.add_argument("--only", default=None, help="episode_id 부분 문자열 필터")
    args = parser.parse_args()

    cfg = load_collection_config(args.config)
    specs = expand_matrix(cfg)
    if args.only:
        specs = [s for s in specs if args.only in s.episode_id]
    todo = [s for s in specs if not is_complete(cfg.output_dir / s.episode_id)]
    if args.limit is not None:
        todo = todo[:args.limit]

    frames = len(todo) * cfg.frames_per_episode
    sim_min = len(todo) * (cfg.warmup_seconds + cfg.frames_per_episode * cfg.capture_interval) / 60
    print(f"[plan] dataset={cfg.name} 전체 {len(specs)}개 중 수집할 에피소드 {len(todo)}개 → {cfg.output_dir}")
    print(f"       프레임 {frames}장, 시뮬레이션 {sim_min:.0f}분, 디스크 약 {frames * EST_MB_PER_FRAME / 1024:.1f}GB")
    for s in todo:
        print(f"       - {s.episode_id}  (seed {s.seed}, NPC {cfg.traffic[s.traffic]}, 보행자 {cfg.walker_count(s.traffic)})")
    if args.dry_run or not todo:
        return

    sim_cfg = load_sim_config()
    sim_cfg.fixed_delta_seconds = cfg.fixed_delta_seconds
    sim_cfg.spawn_index = None
    done = 0
    for map_name, group in itertools.groupby(todo, key=lambda s: s.map):
        sim_cfg.map = map_name
        client, world = connect(sim_cfg)
        with synchronous_mode(world, cfg.fixed_delta_seconds), \
                traffic_manager(client, TM_PORT, cfg.seed) as tm:
            for spec in group:
                done += 1
                print(f"[{done}/{len(todo)}] {spec.episode_id} ...", flush=True)
                stats = run_episode(client, world, tm, cfg, spec, sim_cfg)
                print(f"        {stats['frames']}장 저장 (노란불 추가 {stats['yellow']}), NPC {stats['npc']}대, "
                      f"보행자 {stats['walkers']}명, {stats['wall']:.0f}초 "
                      f"({stats['frames'] / stats['wall']:.1f} 장/초)", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
