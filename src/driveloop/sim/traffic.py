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
