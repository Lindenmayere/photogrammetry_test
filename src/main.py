"""
Test for photogrammetry.

python main.py --images photos/ --out runs/trial01 --true-mm 120 80 45
--true-mm takes the object's three caliper dimensions
"""

import argparse
import csv
import datetime as dt
import shutil
from pathlib import Path
 
import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image, ImageOps
 
IMG_EXTS = {".jpg", ".jpeg", ".png"}

#images, downscale + convert to rgb
def prepare_images(src: Path, dst: Path, max_side: int) -> int:
    """Copy photos to dst, downscaled so the longest side is <= max_side."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(dst / f"{p.stem}.jpg", quality=95)
        n += 1
    return n
 
 
# reconstruct
def reconstruct(image_dir: Path, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    if db.exists():
        db.unlink()
    sparse = work_dir / "sparse"
    sparse.mkdir(exist_ok=True)
 
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE)
    pycolmap.match_exhaustive(db)
    recs = pycolmap.incremental_mapping(db, image_dir, sparse)
    if not recs:
        return None
    return max(recs.values(), key=lambda r: r.num_reg_images())
 
 
def get_points(rec) -> np.ndarray:
    return np.array([p.xyz for p in rec.points3D.values()], dtype=float)
 
 
# measure
def isolate_object(points: np.ndarray, crop_frac: float) -> np.ndarray:
    """
    Rough object isolation for a sparse cloud.
    1) drop statistical outliers
    2) keep points within crop_frac * (90th-percentile distance) of the median
    This is a heuristic: ALWAYS look at the saved object.ply (or use --view)
    and adjust --crop-frac if background points are included or object is cut.
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pts = np.asarray(pcd.points)
    centre = np.median(pts, axis=0)
    d = np.linalg.norm(pts - centre, axis=1)
    keep = d <= crop_frac * np.percentile(d, 90)
    return pts[keep]
 
 
def box_extents(points: np.ndarray) -> np.ndarray:
    """Oriented-bounding-box side lengths (model units), sorted large to small."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    obb = pcd.get_oriented_bounding_box()
    return np.sort(np.asarray(obb.extent))[::-1]
 
 
def scale_and_check(extents: np.ndarray, true_mm):
    """Scale so the largest side matches the true largest side; report errors on all."""
    true_mm = np.sort(np.asarray(true_mm, dtype=float))[::-1]
    scale = true_mm[0] / extents[0]
    measured = extents * scale
    err = measured - true_mm
    pct = 100 * err / true_mm
    return scale, measured, true_mm, err, pct
 
 
# log results
def append_log(args, n_imgs, n_reg, n_pts, reproj, w_err, h_err):
    """Append one row to trials.csv (failed runs are logged too, as FAILED)."""
    log = Path("trials.csv")
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "trial", "images", "registered", "points", "reproj_px",
                        "crop_frac", "W_err_pct", "H_err_pct", "note"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), args.out.name, n_imgs, n_reg,
                    n_pts, reproj, args.crop_frac, w_err, h_err, args.note])
 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path, help="folder of photos")
    ap.add_argument("--out", required=True, type=Path, help="output folder for this trial")
    ap.add_argument("--true-mm", required=True, nargs=3, type=float,
                    metavar=("A", "B", "C"), help="caliper dimensions in mm")
    ap.add_argument("--max-side", type=int, default=2000, help="downscale photos to this size")
    ap.add_argument("--crop-frac", type=float, default=1.0,
                    help="object isolation radius factor (smaller = tighter crop)")
    ap.add_argument("--note", default="", help="free-text note saved to trials.csv")
    ap.add_argument("--view", action="store_true", help="open a 3D viewer of the isolated object")
    args = ap.parse_args()
 
    args.out.mkdir(parents=True, exist_ok=True)
    small = args.out / "images_small"
    n_imgs = prepare_images(args.images, small, args.max_side)
    print(f"Prepared {n_imgs} images")
    if n_imgs < 15:
        print("WARNING: fewer than ~15 photos. Reconstruction may fail.")
 
    rec = reconstruct(small, args.out / "colmap")
    if rec is None:
        print("Reconstruction FAILED (no model). Retake photos with more overlap and texture.")
        return
 
    n_reg = rec.num_reg_images()
    pts = get_points(rec)
    reproj = rec.compute_mean_reprojection_error()
    print(f"Registered {n_reg}/{n_imgs} images, {len(pts)} sparse points, "
          f"mean reprojection error {reproj:.3f} px")
 
    # ts not valid
    min_reg = max(10, int(0.6 * n_imgs))
    if n_reg < min_reg:
        print(f"\nRECONSTRUCTION FAILED: only {n_reg}/{n_imgs} images registered "
              f"(need at least {min_reg}). Dimension numbers would be meaningless, so none are shown.")
        print("Common causes: plain/textureless object, plain background, too little overlap, "
              "blurry or tiny images. Try an object with surface detail on a patterned background.")
        append_log(args, n_imgs, n_reg, len(pts), f"{reproj:.3f}", "FAILED", "FAILED")
        return
 
    obj = isolate_object(pts, args.crop_frac)
    if len(obj) < 30:
        print("Too few points left after isolation; try a larger --crop-frac.")
        return
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj))
    o3d.io.write_point_cloud(str(args.out / "object.ply"), pcd)
 
    extents = box_extents(obj)
    scale, measured, true_sorted, err, pct = scale_and_check(extents, args.true_mm)
 
    print("\nDimension check (largest side used for scaling, so its error is 0 by construction)")
    print(f"{'side':>6} {'true mm':>9} {'model mm':>9} {'err mm':>8} {'err %':>7}")
    for i, name in enumerate(["L", "W", "H"]):
        print(f"{name:>6} {true_sorted[i]:9.1f} {measured[i]:9.1f} {err[i]:8.2f} {pct[i]:7.2f}")
    print("\nThe W and H errors are the real result. Look at object.ply to confirm the crop is sane.")
 
    append_log(args, n_imgs, n_reg, len(pts), f"{reproj:.3f}", f"{pct[1]:.2f}", f"{pct[2]:.2f}")
 
    if args.view:
        o3d.visualization.draw_geometries([pcd])
 
 
if __name__ == "__main__":
    main()