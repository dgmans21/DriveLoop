"""0: 환경 점검 — 서버 접속, 버전 일치, 맵 정보 출력.

    python scripts/00_check_env.py            # 점검
    python scripts/00_check_env.py --reset    # 스크립트 강제 종료 후: 동기 모드 해제 + 남은 차량/센서 삭제
"""
import argparse
import sys

try:
    import carla
except ImportError:
    sys.exit("carla 패키지가 없습니다. `pip install -e .[dev]` 를 먼저 실행하세요.")

from driveloop.config import load_sim_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="동기 모드 해제 + 남은 차량/센서 삭제")
    args = parser.parse_args()

    cfg = load_sim_config()
    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(cfg.timeout)
    try:
        world = client.get_world()
    except RuntimeError as e:
        sys.exit(f"서버 접속 실패 ({cfg.host}:{cfg.port}): {e}\nCARLA 서버가 떠 있는지 확인하세요.")

    print(f"python : {sys.version.split()[0]}")
    print(f"client : {client.get_client_version()}")
    print(f"server : {client.get_server_version()}")
    print(f"map    : {world.get_map().name}")
    towns = sorted(m.split('/')[-1] for m in client.get_available_maps())
    print(f"maps   : {', '.join(towns)}")
    settings = world.get_settings()
    print(f"sync   : {settings.synchronous_mode}  dt={settings.fixed_delta_seconds}")
    print(f"lights : {len(world.get_actors().filter('traffic.traffic_light'))}")
    print(f"spawns : {len(world.get_map().get_spawn_points())}")

    if args.reset:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print("-> 비동기 모드로 복구했습니다.")
        leftovers = [a for a in world.get_actors() if a.type_id.startswith(("vehicle.", "sensor."))]
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in leftovers])
        print(f"-> 남아 있던 차량/센서 {len(leftovers)}개를 삭제했습니다.")

    if client.get_client_version() != client.get_server_version():
        print("\n[!] client/server 버전 불일치 — pip carla 버전을 서버와 맞추세요.")
    else:
        print("\nOK")


if __name__ == "__main__":
    main()
