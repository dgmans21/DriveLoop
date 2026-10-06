"""2-2a 확인용: 라벨 박스를 이미지에 그려 outputs/label_preview/ 에 저장.

    python scripts/21_preview_labels.py --dataset v1 --per-episode 3
    python scripts/21_preview_labels.py --dataset v1 --cls tl_yellow     # 특정 클래스가 있는 프레임만

신호등 박스는 작아서 옆에 확대 이미지를 붙인다.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from driveloop.config import PROJECT_ROOT

COLORS = {"vehicle": (0, 200, 255), "tl_red": (255, 40, 40), "tl_yellow": (255, 220, 0), "tl_green": (40, 255, 80)}


def render(rgb_path: Path, objects: list[dict], title: str, font) -> Image.Image:
    im = Image.open(rgb_path).convert("RGB")
    d = ImageDraw.Draw(im)
    crops = []
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        color = COLORS[o["cls"]]
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if o["cls"] == "vehicle":
            d.text((x1, max(0, y1 - 16)), f"{o['depth']:.0f}m", fill=color, font=font)
        else:
            pad = 6
            crop = Image.open(rgb_path).convert("RGB").crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad))
            scale = 120 / max(1, crop.height)
            crops.append((crop.resize((max(1, int(crop.width * scale)), 120), Image.NEAREST), color, o))
    d.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
    d.text((8, 4), title, fill=(255, 255, 255), font=font)
    if not crops:
        return im
    strip_w = sum(c.width + 10 for c, _, _ in crops) + 10
    out = Image.new("RGB", (max(im.width, strip_w), im.height + 150), (30, 30, 30))
    out.paste(im, (0, 0))
    x = 10
    dd = ImageDraw.Draw(out)
    for crop, color, o in crops:
        out.paste(crop, (x, im.height + 24))
        dd.rectangle([x - 2, im.height + 22, x + crop.width + 1, im.height + 145], outline=color, width=2)
        dd.text((x, im.height + 2), f"{o['depth']:.0f}m {o['facing_angle']:.0f}°", fill=color, font=font)
        x += crop.width + 10
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v1")
    parser.add_argument("--per-episode", type=int, default=2)
    parser.add_argument("--cls", default=None, help="이 클래스가 있는 프레임만")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = PROJECT_ROOT / "data" / "raw" / args.dataset
    labels = PROJECT_ROOT / "data" / "processed" / args.dataset / "labels"
    out = PROJECT_ROOT / "outputs" / "label_preview"
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("arial.ttf", 16)
    rng = random.Random(args.seed)
    saved = 0
    for lab in sorted(labels.glob("*.jsonl")):
        frames = [json.loads(l) for l in lab.open(encoding="utf-8")]
        if args.cls:
            frames = [f for f in frames if any(o["cls"] == args.cls for o in f["objects"])]
        else:  # 신호등이 있는 프레임을 우선
            with_tl = [f for f in frames if any(o["cls"] != "vehicle" for o in f["objects"])]
            frames = with_tl or frames
        for f in rng.sample(frames, min(args.per_episode, len(frames))):
            title = f"{lab.stem} #{f['index']}  " + "  ".join(
                f"{c}:{sum(o['cls'] == c for o in f['objects'])}" for c in COLORS)
            img = render(raw / lab.stem / f["rgb"], f["objects"], title, font)
            img.save(out / f"{lab.stem}_{f['index']:06d}.jpg", quality=90)
            saved += 1
    print(f"[preview] {saved} images → {out}")


if __name__ == "__main__":
    main()
