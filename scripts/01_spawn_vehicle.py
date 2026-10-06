"""1-1: 맵 로드 → 자차 스폰 → 잠깐 직진. 서버 창의 관전자 시점이 차를 따라간다.

    python scripts/01_spawn_vehicle.py --seconds 10
확인: CARLA 서버 창에서 차가 보이고 앞으로 움직이며, 터미널에 속도가 찍히면 성공.
"""
import argparse
import random

import carla

from driveloop.config import load_sim_config
from driveloop.sim.client import (ActorPool, connect, follow_with_spectator, spawn_ego,
                                  speed_mps, synchronous_mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--throttle", type=float, default=0.4)
    parser.add_argument("--map", default=None, help="sim.yaml의 map 덮어쓰기")
    args = parser.parse_args()

    cfg = load_sim_config()
    if args.map:
        cfg.map = args.map
    client, world = connect(cfg)
    rng = random.Random(cfg.seed)

    with synchronous_mode(world, cfg.fixed_delta_seconds), ActorPool() as pool:
        ego, idx = spawn_ego(world, cfg, rng)
        pool.add(ego)
        print(f"[spawn] {ego.type_id} id={ego.id} spawn_point={idx}")

        steps = int(args.seconds / cfg.fixed_delta_seconds)
        per_sec = int(1 / cfg.fixed_delta_seconds)
        for step in range(steps):
            world.tick()
            ego.apply_control(carla.VehicleControl(throttle=args.throttle))
            follow_with_spectator(world, ego)
            if step % per_sec == 0:
                loc = ego.get_location()
                print(f"t={step * cfg.fixed_delta_seconds:4.1f}s  speed={speed_mps(ego) * 3.6:5.1f} km/h  "
                      f"pos=({loc.x:.1f}, {loc.y:.1f})")
    print("[done] 액터 정리 및 설정 복구 완료")


if __name__ == "__main__":
    main()
