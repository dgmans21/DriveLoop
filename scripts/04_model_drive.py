"""2-4: 내 모델(YOLO)로 인지해서 운전 — 1-3과 판단·제어는 같고 인지만 교체.

    python scripts/04_model_drive.py --perception model --record --seconds 120
    python scripts/04_model_drive.py --perception gt --record --seconds 120     # 비교용 (정답값 인지)
    python scripts/04_model_drive.py --perception model --weights runs/detect/v2_base/weights/best.pt  # 이전 모델과 비교

주의: 모델은 Epic 화질 이미지로 학습했다 → 서버를 Epic으로 띄울 것
      (start_carla.ps1 -Quality Epic -OffScreen). Low 화질이면 처음 보는 이미지라 성능이 크게 떨어진다.

화면/녹화: 전방 카메라(모델 입력) + 검출 박스 + 지도로 예상한 내 신호 위치(흰 테두리) + 고른 검출(MY LIGHT)
          + 오른쪽 아래 3인칭 화면. 녹화는 매 tick(20 fps) = 실제 시간 속도.
로그(CSV): 매 tick 모델 판단과 실제 신호 상태(gt_state)를 나란히 기록 — gt_state는 판단에 쓰지 않고 비교 분석용.
웹(--web): 오버레이 없는 전방·3인칭 영상(H.264) + 프레임별 frames.json → outputs/web/<tag>/
          웹 페이지가 영상 위에 박스·패널을 직접 그린다 (켜고 끄기, 그래프와 시간 동기화, HUD 변경에 재녹화 불필요)

    python scripts/04_model_drive.py --perception model --map Town05 --seconds 180 --record --web --no-display
날씨(--weather clear|rain, --time noon|night): 수집 설정(configs/collection/v3.yaml)과 같은 값 → 학습한 조건 재현.
NPC 차량은 없다: 판단에 앞차 거리 유지가 아직 없어서 (PerceptionOutput의 TODO lead_distance) 추돌한다.
"""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import carla

from driveloop.agent import DrivingAgent
from driveloop.config import CameraConfig, PROJECT_ROOT, load_driving_config, load_sim_config
from driveloop.data.collection_config import CaptureCamera, load_collection_config
from driveloop.sim.client import ActorPool, connect, spawn_ego, synchronous_mode
from driveloop.sim.sensors import attach_rgb_camera, get_frame, image_to_rgb
from driveloop.sim.weather import apply_weather
from driveloop.viz.hud import Display
from driveloop.viz.overlay import STATE_RGB, Mp4Recorder, draw_detections, draw_panel, paste_inset, to_web_mp4

WHITE, GRAY, CYAN = (255, 255, 255), (180, 180, 180), (90, 210, 255)
BEHAVIOR_RGB = {"CRUISE": (80, 230, 110), "CAUTION": (200, 160, 255), "STOPPING": (255, 210, 60), "STOPPED": (255, 70, 70),
                "PROCEED_YELLOW": (90, 210, 255)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", choices=["model", "gt"], default="model")
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "runs" / "detect" / "v3_base" / "weights" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.10,
                        help="검출 신뢰도 임계값. Town05 test 스윕: 0.25→0.10에서 red recall 0.76→0.82, "
                             "늘어난 오검출은 지도 예상 위치 근처만 쓰므로 영향 작음")
    parser.add_argument("--weather", default=None, help="수집 설정의 날씨 이름 (clear / rain). 없으면 맵 기본값")
    parser.add_argument("--time", default=None, help="수집 설정의 시간대 이름 (noon / night)")
    parser.add_argument("--conditions", default=str(PROJECT_ROOT / "configs" / "collection" / "v3.yaml"),
                        help="날씨·시간대 값을 가져올 수집 설정 (학습 데이터와 같은 조건을 재현)")
    parser.add_argument("--map", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=None, help="시뮬레이션 시간 (없으면 ESC까지)")
    parser.add_argument("--record", action="store_true", help="outputs/videos/ 에 MP4 녹화")
    parser.add_argument("--no-display", action="store_true", help="창 없이 실행 (녹화·로그만)")
    parser.add_argument("--web", action="store_true",
                        help="웹 오버레이용 깨끗한 영상 + 프레임별 JSON (outputs/web/<tag>/)")
    args = parser.parse_args()

    cfg = load_sim_config()
    drv = load_driving_config()
    if args.map:
        cfg.map = args.map
    if args.seed is not None:
        cfg.seed = args.seed
    dt = cfg.fixed_delta_seconds
    front = CaptureCamera()  # 학습 데이터와 같은 위치·해상도·시야각

    client, world = connect(cfg)
    cond = "default"
    if args.weather or args.time:
        presets = load_collection_config(args.conditions)
        weather = args.weather or "clear"
        time_of_day = args.time or "noon"
        apply_weather(world, {**presets.weathers[weather], **presets.times[time_of_day]})
        cond = f"{weather}_{time_of_day}"
    rng = random.Random(cfg.seed)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    tag = f"{args.perception}_{cfg.map}_{cond}_s{cfg.seed}_{stamp}"
    out_dir = PROJECT_ROOT / "outputs"
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"drive_{tag}.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    log.writerow(["t", "x", "y", "speed", "perception", "tl_state", "gt_state", "raw_state", "assoc_conf",
                  "stop_dist", "state", "target_speed", "infer_ms", "n_det"])
    display = None if args.no_display else Display(front.width, front.height, f"DriveLoop 2-4: {args.perception}")
    recorder = Mp4Recorder(out_dir / "videos" / f"drive_{tag}.mp4", 1 / dt, (front.width, front.height)) \
        if args.record else None
    web_dir = out_dir / "web" / tag if args.web else None
    web_front = web_chase = None
    web_frames: list[dict] = []
    if web_dir is not None:
        web_dir.mkdir(parents=True, exist_ok=True)
        web_front = Mp4Recorder(web_dir / "front_raw.mp4", 1 / dt, (front.width, front.height))
        web_chase = Mp4Recorder(web_dir / "chase_raw.mp4", 1 / dt, (cfg.camera.width, cfg.camera.height))
    print(f"[log] {log_path}" + (f"\n[rec] {recorder.path}" if recorder else "")
          + (f"\n[web] {web_dir}" if web_dir else ""))

    try:
        with synchronous_mode(world, dt), ActorPool() as pool:
            ego, idx = spawn_ego(world, cfg, rng)
            pool.add(ego)
            if world.get_weather().sun_altitude_angle < 0:
                # 수집 때는 Traffic Manager가 켜 줬다. 직접 조종하는 자차는 전조등을 직접 켠다 (학습 이미지와 같은 조건)
                ego.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position
                                                            | carla.VehicleLightState.LowBeam))
            front_cam, front_q = attach_rgb_camera(
                world, ego, CameraConfig(front.width, front.height, front.fov),
                carla.Transform(carla.Location(x=front.x, z=front.z)))
            pool.add(front_cam)
            chase_cam, chase_q = attach_rgb_camera(world, ego, cfg.camera)
            pool.add(chase_cam)
            world.tick()

            def make_model(route, _gt):
                from driveloop.perception.model import ModelPerception
                return ModelPerception(world, ego, route, front_cam, front.width, front.height, front.fov,
                                       args.weights, drv.tl_lookahead, conf=args.conf,
                                       lead_lookahead=drv.lead_lookahead, lane_half_width=drv.lane_half_width,
                                       cam_height=drv.mono_cam_height, dt=dt)
            agent = DrivingAgent(world, ego, drv, rng, make_model if args.perception == "model" else None)
            perception = agent.perception
            print(f"[start] perception={args.perception} map={cfg.map} spawn={idx}"
                  + (f" weights={args.weights}" if args.perception == "model" else ""))

            prev_state = agent.behavior.state
            t0 = world.get_snapshot().timestamp.elapsed_seconds
            while True:
                if display is not None and display.poll_quit():
                    break
                frame = world.tick()
                front_img = get_frame(front_q, frame)
                chase_img = get_frame(chase_q, frame)
                sim_t = world.get_snapshot().timestamp.elapsed_seconds - t0
                if args.seconds is not None and sim_t >= args.seconds:
                    break

                # 인지 → 판단 → 제어 (판단·제어는 1-3과 동일, agent.py)
                r = agent.step(front_img, dt)
                p, gt_p, decision, target_speed, speed, tf = r.p, r.gt, r.decision, r.target_speed, r.speed, r.transform

                dbg = getattr(perception, "debug", None)
                tl_val = p.tl_state.value if p.tl_state else ""
                gt_val = gt_p.tl_state.value if gt_p.tl_state else ""
                log.writerow([f"{sim_t:.2f}", f"{tf.location.x:.2f}", f"{tf.location.y:.2f}", f"{speed:.2f}",
                              args.perception, tl_val, gt_val,
                              (dbg.raw_state or "") if dbg else "",
                              f"{dbg.association.conf:.2f}" if dbg and dbg.association else "",
                              "" if p.stop_distance is None else f"{p.stop_distance:.2f}",
                              decision.state.value, f"{target_speed:.2f}",
                              f"{dbg.infer_ms:.1f}" if dbg else "", len(dbg.detections) if dbg else ""])
                if decision.state is not prev_state:
                    print(f"[t={sim_t:7.2f}s] {prev_state.value:>14} -> {decision.state.value:<14} "
                          f"v={speed * 3.6:5.1f}km/h  light={tl_val or '-'} (gt {gt_val or '-'})  {decision.reason}")
                    prev_state = decision.state

                if web_dir is not None:
                    web_front.write(image_to_rgb(front_img))
                    web_chase.write(image_to_rgb(chase_img))
                    web_frames.append(web_frame(sim_t, speed, target_speed, decision, p, gt_val, dbg))

                if display is None and recorder is None:
                    continue
                view = image_to_rgb(front_img)
                if dbg is not None:
                    view = draw_detections(view, dbg.detections, dbg.expected,
                                           dbg.association.det if dbg.association else None)
                light = "none" if p.tl_state is None else f"{tl_val} {p.stop_distance:4.1f} m"
                lines = [
                    (f"Perception: {'MY MODEL (YOLO)' if args.perception == 'model' else 'ground truth'}", CYAN),
                    (f"Light  : {light}", STATE_RGB.get(tl_val, GRAY)),
                ]
                if dbg is not None:
                    lines.append((f"  raw {dbg.raw_state or '-':7} | vote {dbg.voted_state:7} | gt {gt_val or '-'}",
                                  GRAY))
                lines += [
                    (f"State  : {decision.state.value}", BEHAVIOR_RGB.get(decision.state.value, WHITE)),
                    (f"Speed  : {speed * 3.6:5.1f} / {target_speed * 3.6:5.1f} km/h", WHITE),
                ]
                if dbg is not None:
                    lines.append((f"Infer  : {dbg.infer_ms:5.1f} ms  ({len(dbg.detections)} det)", GRAY))
                view = draw_panel(view, lines)
                view = paste_inset(view, image_to_rgb(chase_img), 0.28)
                if recorder:
                    recorder.write(view)
                if display is not None:
                    display.draw(view)
    finally:
        log_file.close()
        if recorder:
            recorder.close()
            print(f"[rec] {recorder.frames} frames → {recorder.path}")
        if display is not None:
            display.close()
        if web_dir is not None:
            finish_web(web_dir, web_front, web_chase, web_frames, {
                "map": cfg.map, "seed": cfg.seed, "condition": cond, "perception": args.perception,
                "weights": Path(args.weights).as_posix() if args.perception == "model" else None,
                "conf": args.conf, "fps": round(1 / dt), "width": front.width, "height": front.height,
            })


def web_frame(t, speed, target_speed, decision, p, gt_val, dbg) -> dict:
    """웹 오버레이용 한 프레임. 박스는 정수 px, 키는 짧게 (2분 = 2,400프레임)."""
    rec = {"t": round(t, 2), "v": round(speed * 3.6, 1), "vt": round(target_speed * 3.6, 1),
           "st": decision.state.value, "tl": p.tl_state.value if p.tl_state else None, "gt": gt_val or None,
           "d": None if p.stop_distance is None else round(p.stop_distance, 1)}
    if dbg is not None:
        chosen = dbg.association.det if dbg.association else None
        rec.update({
            "raw": dbg.raw_state, "ms": round(dbg.infer_ms, 1),
            "det": [[d.cls, *[int(round(x)) for x in d.box], round(d.conf, 2)] for d in dbg.detections],
            "exp": [[int(round(x)) for x in b] for b in dbg.expected],
            "pick": next((i for i, d in enumerate(dbg.detections) if d is chosen), None),
        })
    return rec


def finish_web(web_dir: Path, front_rec: Mp4Recorder, chase_rec: Mp4Recorder, frames: list[dict], meta: dict) -> None:
    """영상 닫기 → H.264 변환(브라우저 재생용) → frames.json."""
    front_rec.close()
    chase_rec.close()
    for name, rec in (("front", front_rec), ("chase", chase_rec)):
        if to_web_mp4(rec.path, web_dir / f"{name}.mp4"):
            Path(rec.path).unlink()
        else:
            print(f"[web] ffmpeg 없음 → {rec.path} 는 브라우저 재생 불가 (mp4v)")
    (web_dir / "frames.json").write_text(json.dumps({"meta": meta, "frames": frames}, ensure_ascii=False,
                                                    separators=(",", ":")), encoding="utf-8")
    print(f"[web] {len(frames)} frames → {web_dir}")


if __name__ == "__main__":
    main()
