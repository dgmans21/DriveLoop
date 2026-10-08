"""모델 검출 결과 영상 (원본 프레임 위에 예측 박스).

    python scripts/32_detection_video.py --run v2_base --source v2 \
        --episodes Town05_clear_noon_high_r0 Town05_clear_night_high_r0 Town05_rain_noon_high_r0 Town05_rain_night_high_r0

수집 프레임은 0.5초 간격(노란불 구간 0.25초)이라 연속 영상이 아니라 타임랩스처럼 보인다.
부드러운 영상은 2-4에서 CARLA 주행을 매 프레임 녹화해 만든다.
결과: outputs/videos/detect_<run>_<source>.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from driveloop.config import PROJECT_ROOT
from driveloop.viz.overlay import _text

# BGR. 차량=파랑, 신호 색=같은 색 (분포 리포트와 동일한 팔레트)
COLORS = {"vehicle": (214, 120, 42), "tl_red": (72, 73, 227), "tl_yellow": (0, 161, 237), "tl_green": (0, 131, 0)}
LABEL = {"vehicle": "car", "tl_red": "red", "tl_yellow": "yellow", "tl_green": "green"}


def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    _text(img, text, org, scale, color, thick)     # 테두리 잔상 없는 공용 함수 (viz/overlay.py)


def title_card(w, h, lines):
    img = np.full((h, w, 3), 24, np.uint8)
    for i, t in enumerate(lines):
        put_text(img, t, (60, h // 2 - 30 + i * 50), 1.1 if i == 0 else 0.8)
    return img


def draw(img, names, result, header):
    counts = {}
    for (x1, y1, x2, y2), c, s in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        cls = names[int(c)]
        counts[cls] = counts.get(cls, 0) + 1
        col = COLORS[cls]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        put_text(img, f"{LABEL[cls]} {s:.2f}", (int(x1), max(14, int(y1) - 5)), 0.45, col, 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    put_text(img, header, (10, 25), 0.65)
    x = img.shape[1] - 420
    for cls in COLORS:
        put_text(img, f"{LABEL[cls]}:{counts.get(cls, 0)}", (x, 25), 0.6, COLORS[cls], 2)
        x += 105
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="v2_base")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--source", default="v2")
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(PROJECT_ROOT / "runs" / "detect" / args.run / "weights" / args.weights))
    names = model.names
    out_path = PROJECT_ROOT / "outputs" / "videos" / f"detect_{args.run}_{args.source}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total = 0
    for ep in args.episodes:
        ep_dir = PROJECT_ROOT / "data" / "raw" / args.source / ep
        meta = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
        frames = [json.loads(l) for l in (ep_dir / "frames.jsonl").open(encoding="utf-8")]
        W, H = meta["camera"]["width"], meta["camera"]["height"]
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        tags = meta["tags"]
        card = title_card(W, H, [f"{tags['map']}  {tags['weather']} / {tags['time']} / traffic {tags['traffic']}",
                                 f"model: {args.run} ({args.weights})  -  unseen test map",
                                 "frames captured every 0.5 s (0.25 s while yellow) -> time-lapse"])
        for _ in range(int(args.fps * 1.5)):
            writer.write(card)
        paths = [str(ep_dir / f["rgb"]) for f in frames]
        for i in range(0, len(paths), args.batch):
            results = model.predict(paths[i:i + args.batch], conf=args.conf, imgsz=1280, verbose=False)
            for r, f in zip(results, frames[i:i + args.batch]):
                header = (f"{tags['map']} {tags['weather']}/{tags['time']}  t={f['t']:5.1f}s  "
                          f"ego {f['ego']['speed'] * 3.6:4.1f} km/h")
                writer.write(draw(r.orig_img.copy(), names, r, header))
                total += 1
        print(f"  {ep}: {len(frames)} frames")
    writer.release()
    print(f"[video] {total} frames, {total / args.fps:.0f}s @ {args.fps}fps → {out_path}")


if __name__ == "__main__":
    main()
