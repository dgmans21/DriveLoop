"""1-2: 카메라 부착 → pygame 화면 표시. 키보드로 직접 운전.

    python scripts/02_camera_view.py
조작: W/↑ 가속, S/↓ 브레이크, A/D(←/→) 조향, ESC 종료
확인: pygame 창에 3인칭 시점 영상이 나오고, 키 입력대로 차가 움직이면 성공.
"""
import random

import carla
import pygame

from driveloop.config import load_sim_config
from driveloop.sim.client import ActorPool, connect, spawn_ego, speed_mps, synchronous_mode
from driveloop.sim.sensors import attach_rgb_camera, get_frame, image_to_rgb
from driveloop.viz.hud import CYAN, WHITE, Display


def keyboard_control() -> carla.VehicleControl:
    keys = pygame.key.get_pressed()
    throttle = 0.6 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0
    brake = 1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0
    steer = 0.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer -= 0.4
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer += 0.4
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


def main() -> None:
    cfg = load_sim_config()
    client, world = connect(cfg)
    rng = random.Random(cfg.seed)
    display = Display(cfg.camera.width, cfg.camera.height, "DriveLoop 1-2: camera")

    try:
        with synchronous_mode(world, cfg.fixed_delta_seconds), ActorPool() as pool:
            ego, _ = spawn_ego(world, cfg, rng)
            pool.add(ego)
            camera, cam_q = attach_rgb_camera(world, ego, cfg.camera)
            pool.add(camera)

            while not display.poll_quit():
                frame = world.tick()
                image = get_frame(cam_q, frame)
                ego.apply_control(keyboard_control())
                display.draw(image_to_rgb(image), [
                    (f"Speed : {speed_mps(ego) * 3.6:5.1f} km/h", WHITE),
                    (f"Frame : {frame}", WHITE),
                    (f"FPS   : {display.fps:4.1f}", CYAN),
                ])
    finally:
        display.close()


if __name__ == "__main__":
    main()
