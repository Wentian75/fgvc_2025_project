#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--max-categories", type=int, default=5)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    cats = data["categories"]

    # pick first K categories by id ordering
    cats_sorted = sorted(cats, key=lambda c: c["id"])[: args.max_categories]
    allowed_cat_ids = {c["id"] for c in cats_sorted}

    anns = [a for a in data["annotations"] if a["category_id"] in allowed_cat_ids]

    # collect images referenced by anns, up to max-images unique
    img_ids = []
    seen = set()
    for a in anns:
        iid = a["image_id"]
        if iid in seen:
            continue
        img_ids.append(iid)
        seen.add(iid)
        if len(img_ids) >= args.max_images:
            break

    allowed_img_ids = set(img_ids)
    images = [im for im in data["images"] if im["id"] in allowed_img_ids]
    ann_filtered = [a for a in anns if a["image_id"] in allowed_img_ids]

    out = {
        "info": data.get("info", {}),
        "images": images,
        "annotations": ann_filtered,
        "categories": cats_sorted,
        "licenses": data.get("licenses", []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote tiny COCO: images={len(images)}, anns={len(ann_filtered)}, cats={len(cats_sorted)} to {args.output}")


if __name__ == "__main__":
    main()

