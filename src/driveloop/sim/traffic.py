"""Traffic Manager 기반 NPC 차량."""
from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator

import carla

NPC_BASE_TYPES = {"car", "van", "truck", "bus", "motorcycle"}  # 자전거 제외


@contextmanager
def traffic_manager(client: carla.Client, port: int, seed: int) -> Iterator[carla.TrafficManager]:
    """동기 모드 TM. world 동기 모드 안에서 사용하고, 종료 시 동기 모드를 해제한다."""
    tm = client.get_trafficmanager(port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    try:
        yield tm
    finally:
        tm.set_synchronous_mode(False)


@contextmanager
def traffic_light_timing(world: carla.World, green: float, yellow: float, red: float) -> Iterator[None]:
    """에피소드 동안만 모든 신호등 시간을 바꾸고, 끝나면 원래 값으로 되돌린다 (다음 수집에 남지 않게)."""
    lights = list(world.get_actors().filter("traffic.traffic_light"))
    original = [(tl, tl.get_green_time(), tl.get_yellow_time(), tl.get_red_time()) for tl in lights]
    for tl in lights:
        tl.set_green_time(green)
        tl.set_yellow_time(yellow)
        tl.set_red_time(red)
    world.reset_all_traffic_lights()
    try:
        yield
    finally:
        for tl, g, y, r in original:
            tl.set_green_time(g)
            tl.set_yellow_time(y)
            tl.set_red_time(r)
        world.reset_all_traffic_lights()


def spawn_npc_vehicles(client: carla.Client, world: carla.World, tm: carla.TrafficManager,
                       count: int, rng: random.Random) -> list[int]:
    lib = world.get_blueprint_library()
    blueprints = [bp for bp in lib.filter("vehicle.*")
                  if bp.has_attribute("base_type")
                  and bp.get_attribute("base_type").as_str().lower() in NPC_BASE_TYPES]
    points = world.get_map().get_spawn_points()
    rng.shuffle(points)

    batch = []
    for tf in points[:count]:
        bp = rng.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        batch.append(carla.command.SpawnActor(bp, tf)
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
    results = client.apply_batch_sync(batch, True)
    ids = [r.actor_id for r in results if not r.error]
    for actor in world.get_actors(ids):
        tm.update_vehicle_lights(actor, True)  # 야간 전조등 자동
    return ids
