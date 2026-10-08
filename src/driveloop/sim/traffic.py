"""Traffic Manager 기반 NPC 차량 + AI 보행자."""
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


def spawn_npc_walkers(client: carla.Client, world: carla.World, count: int, rng: random.Random,
                      cross_factor: float = 0.3, running: float = 0.1, seed: int = 0) -> tuple[list[int], list[int]]:
    """AI 보행자 → (보행자 id, 컨트롤러 id). 보도(내비게이션 메시) 위 무작위 위치에서 무작위 목적지로 걷는다.

    cross_factor: 차도를 건너는 비율 (CARLA 기본 0 → 보도만 걸어 '차도 위 보행자' 장면이 거의 없다).
    정리할 때는 컨트롤러를 stop() 한 뒤 파괴해야 한다 (ActorPool이 처리).
    """
    if count <= 0:
        return [], []
    world.set_pedestrians_seed(seed)                       # 내비게이션 무작위 위치·경로 재현
    world.set_pedestrians_cross_factor(cross_factor)
    blueprints = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    batch, speeds = [], []
    for _ in range(count):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = rng.choice(blueprints)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        run = rng.random() < running
        sp = bp.get_attribute("speed").recommended_values if bp.has_attribute("speed") else ["1.4", "1.4", "3.0"]
        speeds.append(float(sp[2] if run and len(sp) > 2 else sp[1]))
        batch.append(carla.command.SpawnActor(bp, carla.Transform(loc + carla.Location(z=1.0))))
    results = client.apply_batch_sync(batch, True)
    walker_ids, walker_speeds = [], []
    for r, v in zip(results, speeds):
        if not r.error:
            walker_ids.append(r.actor_id)
            walker_speeds.append(v)
    ctrl_bp = world.get_blueprint_library().find("controller.ai.walker")
    results = client.apply_batch_sync([carla.command.SpawnActor(ctrl_bp, carla.Transform(), wid)
                                       for wid in walker_ids], True)
    pairs = [(wid, r.actor_id, v) for wid, r, v in zip(walker_ids, results, walker_speeds) if not r.error]
    world.tick()                                           # 컨트롤러가 붙은 뒤에 start 해야 한다
    for _, cid, v in pairs:
        ctrl = world.get_actor(cid)
        ctrl.start()
        ctrl.go_to_location(world.get_random_location_from_navigation())
        ctrl.set_max_speed(v)
    controller_ids = {cid for _, cid, _ in pairs}
    orphans = [wid for wid in walker_ids if wid not in {w for w, _, _ in pairs}]
    if orphans:                                            # 컨트롤러가 안 붙은 보행자는 서 있기만 함 → 제거
        client.apply_batch_sync([carla.command.DestroyActor(w) for w in orphans], True)
    return [w for w, _, _ in pairs], sorted(controller_ids)
