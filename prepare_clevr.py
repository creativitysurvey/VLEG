"""
Builds a LARGE-SCALE single-object CLEVR crop set for the real-image
VLEG pipeline: 16 attribute combinations x ~32 distinct instances each
(~512 total images), closer to the paper's originally-specified
500-query scale than the earlier 16-image (1 per combo) pilot version.

Attribute mapping (see DEVIATIONS_LOG.md for full rationale):
  shape    in {cube, sphere}   -- causal
  color    in {red, blue}      -- causal
  material in {rubber, metal}  -- background-analog, flips the rule
  size     in {large, small}   -- distractor, non-causal

Crop method: same isolation-distance ranking as the earlier pilot
version (picks the instance whose center is farthest from any other
object's center in the same scene, to avoid capturing neighbors) --
but now keeps the TOP N candidates per combo instead of just the single
best one, to get many distinct real instances of the same attribute
combination. This lets train/test splits use genuinely different
images of "a red rubber cube" rather than the exact same image, testing
real instance-level generalization, not memorization of one crop.

Run:
    python prepare_clevr.py --clevr-root ~/myshare/CLEVR_v1.0 --split train --n-per-combo 32
"""
import argparse
import json
import math
from pathlib import Path
from PIL import Image

TARGET_SHAPES = ["cube", "sphere"]
TARGET_COLORS = ["red", "blue"]
TARGET_MATERIALS = ["rubber", "metal"]
TARGET_SIZES = ["large", "small"]

CROP_SIZE = 180  # pixels, square crop centered on the object
MIN_ISOLATION_PX = 140  # candidates below this are flagged, not automatically excluded


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def find_candidates(clevr_root: Path, split: str):
    scenes_path = clevr_root / "scenes" / f"CLEVR_{split}_scenes.json"
    print(f"Loading {scenes_path} ...")
    data = json.loads(scenes_path.read_text())
    scenes = data["scenes"]
    print(f"Loaded {len(scenes)} scenes.")

    candidates = {}
    for scene in scenes:
        objs = scene["objects"]
        n_objs = len(objs)
        for i, obj in enumerate(objs):
            shape, color, material, size = obj["shape"], obj["color"], obj["material"], obj["size"]
            if shape not in TARGET_SHAPES or color not in TARGET_COLORS:
                continue
            if material not in TARGET_MATERIALS or size not in TARGET_SIZES:
                continue
            others = [o["pixel_coords"] for j, o in enumerate(objs) if j != i]
            isolation = min((dist(obj["pixel_coords"], o) for o in others), default=1e9)
            combo = (shape, color, material, size)
            candidates.setdefault(combo, []).append(
                (isolation, n_objs, scene["image_filename"], obj["pixel_coords"])
            )
    return candidates


def crop_and_save(clevr_root: Path, split: str, image_filename: str,
                   pixel_coords, out_path: Path):
    img_path = clevr_root / "images" / split / image_filename
    img = Image.open(img_path)
    cx, cy = pixel_coords[0], pixel_coords[1]
    half = CROP_SIZE // 2
    left, top = max(0, cx - half), max(0, cy - half)
    right, bottom = left + CROP_SIZE, top + CROP_SIZE
    if right > img.width:
        right, left = img.width, img.width - CROP_SIZE
    if bottom > img.height:
        bottom, top = img.height, img.height - CROP_SIZE
    crop = img.crop((left, top, right, bottom))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clevr-root", required=True, help="Path to extracted CLEVR_v1.0/")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out-dir", default="clevr_crops_large")
    ap.add_argument("--n-per-combo", type=int, default=32,
                     help="How many distinct instances to keep per attribute combo.")
    args = ap.parse_args()

    clevr_root = Path(args.clevr_root)
    out_dir = Path(args.out_dir)

    candidates = find_candidates(clevr_root, args.split)

    print(f"\nFound candidates for {len(candidates)}/16 target combos.")
    index = []
    for shape in TARGET_SHAPES:
        for color in TARGET_COLORS:
            for material in TARGET_MATERIALS:
                for size in TARGET_SIZES:
                    combo = (shape, color, material, size)
                    cands = candidates.get(combo, [])
                    if len(cands) < args.n_per_combo:
                        print(f"  WARNING: {combo} only has {len(cands)} candidates "
                              f"(< requested {args.n_per_combo})")
                    cands.sort(key=lambda c: c[0], reverse=True)  # most isolated first
                    chosen = cands[:args.n_per_combo]
                    n_low_isolation = sum(1 for c in chosen if c[0] < MIN_ISOLATION_PX)

                    instances = []
                    for idx, (isolation, n_objs, image_filename, pixel_coords) in enumerate(chosen, start=1):
                        out_name = f"{shape}_{color}_{material}_{size}_{idx:02d}.png"
                        out_path = out_dir / out_name
                        crop_and_save(clevr_root, args.split, image_filename, pixel_coords, out_path)
                        instances.append({
                            "image_path": str(out_path), "source_image": image_filename,
                            "source_n_objects": n_objs, "isolation_px": round(isolation, 1),
                        })
                    index.append({
                        "shape": shape, "color": color, "material": material, "size": size,
                        "instances": instances,
                    })
                    flag = f"  ** {n_low_isolation} low-isolation instance(s), spot-check **" if n_low_isolation else ""
                    print(f"  OK: {combo} -> {len(instances)} instances cropped{flag}")

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    total_images = sum(len(e["instances"]) for e in index)
    print(f"\n{len(index)}/16 combos resolved, {total_images} total images cropped. "
          f"Index written to {index_path}")


if __name__ == "__main__":
    main()
