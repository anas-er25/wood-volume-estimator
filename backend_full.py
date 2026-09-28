"""
WOOD VOLUME ESTIMATOR V5
========================

3 photos -> 2D detection -> multi-view geometry -> sparse 3D reconstruction.

V5 replaces the previous "bbox * tire scale" approach with:
    1. truck detection
    2. automatic loading-zone proposal
    3. experimental wood-load segmentation
    4. SIFT feature extraction
    5. pairwise feature matching
    6. essential-matrix / relative-camera estimation
    7. sparse triangulation
    8. optional metric scale estimation from a detected tire
    9. conservative 3D diagnostics

IMPORTANT:
- A normal tire diameter is only a reference assumption. It is not a complete
  camera calibration.
- Three independent AI-generated images are NOT a valid photogrammetry set.
  The same physical scene must be photographed from different viewpoints.
- V5 intentionally returns "not computable" if the geometry is insufficient.
- Sparse point-cloud volume is NOT the same as a watertight wood volume.
- A future V6 should use COLMAP/Open3D + trained wood segmentation for
  production-grade reconstruction.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception:
    OPEN3D_AVAILABLE = False


# Configuration
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n-seg.pt")
REFERENCE_TIRE_DIAMETER_M = float(os.getenv("REFERENCE_TIRE_DIAMETER_M", "1.00"))
MIN_MATCHES = int(os.getenv("MIN_MATCHES", "40"))
MIN_INLIERS = int(os.getenv("MIN_INLIERS", "20"))
FOCAL_RATIO = float(os.getenv("FOCAL_RATIO", "1.20"))
COMPACTION_FACTOR = float(os.getenv("COMPACTION_FACTOR", "0.70"))


# App
app = FastAPI(
    title="Wood Volume Estimator V5",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# Models
try:
    yolo_model = YOLO(YOLO_MODEL_NAME)
except Exception as exc:
    yolo_model = None
    print(f"[WARN] YOLO unavailable: {exc}")


# JSON helpers
def py(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    return value


def safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


# Image processing
def decode_image(data: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def resize_for_geometry(image: np.ndarray, max_width: int = 1600) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if w <= max_width:
        return image, 1.0
    scale = max_width / float(w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


# Tire detection
def detect_tire(image: np.ndarray) -> Optional[Dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    h, w = gray.shape
    min_r = max(15, int(min(h, w) * 0.035))
    max_r = max(min_r + 5, int(min(h, w) * 0.22))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, int(min(h, w) * 0.10)),
        param1=100,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return None

    candidates = []
    for x, y, r in np.round(circles[0]).astype(int):
        if r <= 0:
            continue
        x1 = max(0, x-r)
        y1 = max(0, y-r)
        x2 = min(w, x+r)
        y2 = min(h, y+r)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        mean = float(np.mean(roi))
        score = r * 2.0 + max(0.0, 120.0 - mean)
        candidates.append((score, x, y, r, mean))

    if not candidates:
        return None

    _, x, y, r, mean = max(candidates, key=lambda z: z[0])
    return {
        "cx_px": float(x),
        "cy_px": float(y),
        "radius_px": float(r),
        "diameter_px": float(2*r),
        "reference_diameter_m": float(REFERENCE_TIRE_DIAMETER_M),
    }


# Truck detection
def detect_truck(image: np.ndarray) -> Optional[Dict[str, Any]]:
    if yolo_model is None:
        return None

    try:
        results = yolo_model.predict(image, verbose=False, conf=0.20)
    except Exception as exc:
        print(f"[WARN] YOLO failed: {exc}")
        return None

    best = None
    for result in results:
        if result.boxes is None:
            continue
        names = result.names
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, class_id)).lower()
            if class_name != "truck":
                continue
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(float)
            area = max(1.0, (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))
            candidate = (conf, area, result, i, bbox)
            if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] > best[1]):
                best = candidate

    if best is None:
        return None

    conf, _, result, index, bbox = best
    mask = None
    if getattr(result, "masks", None) is not None and result.masks.data is not None:
        try:
            mask = result.masks.data[index].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = mask > 0
        except Exception:
            mask = None

    return {
        "confidence": float(conf),
        "bbox": [float(x) for x in bbox],
        "mask_available": mask is not None,
        "mask": mask,
    }


# Loading zone
def detect_loading_zone(image: np.ndarray, truck: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if truck is None:
        return {"bbox": None, "confidence": 0.0}

    h, w = image.shape[:2]
    x1, y1, x2, y2 = truck["bbox"]
    bw = max(1.0, x2-x1)
    bh = max(1.0, y2-y1)

    lx1 = int(max(0, x1 + 0.18*bw))
    lx2 = int(min(w, x2 - 0.03*bw))
    ly1 = int(max(0, y1 + 0.08*bh))
    ly2 = int(min(h, y1 + 0.76*bh))

    return {
        "bbox": [lx1, ly1, max(lx1+1, lx2), max(ly1+1, ly2)],
        "confidence": float(np.clip(0.45 + 0.45*truck["confidence"], 0, 0.95)),
    }


# Wood segmentation
def detect_wood(image: np.ndarray, loading: Dict[str, Any]) -> Dict[str, Any]:
    bbox = loading["bbox"]
    if bbox is None:
        return {"fill_ratio": None, "confidence": 0.0, "method": "unavailable"}

    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"fill_ratio": None, "confidence": 0.0, "method": "empty"}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = cv2.GaussianBlur(np.abs(lap), (9, 9), 0)
    threshold = max(5.0, float(np.percentile(texture, 55)))
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    candidate = (texture >= threshold) & (value > 30) & (value < 250) & (saturation > 15)
    mask = candidate.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    fill = float(np.mean(mask > 0))
    confidence = float(np.clip(0.25 + 0.55*min(1.0, fill*2), 0, 0.80))

    return {"fill_ratio": fill, "confidence": confidence, "method": "heuristic_texture_segmentation"}


# Feature matching
def create_features(image: np.ndarray, roi: Optional[List[int]] = None) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if roi:
        x1, y1, x2, y2 = roi
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
    else:
        mask = None

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    return keypoints, descriptors


def match_features(kp1, des1, kp2, des2) -> List[cv2.DMatch]:
    if des1 is None or des2 is None:
        return []
    if len(des1) < 8 or len(des2) < 8:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.72*n.distance:
            good.append(m)
    return good


# Pairwise geometry
def estimate_pair_geometry(image1: np.ndarray, image2: np.ndarray, kp1, des1, kp2, des2) -> Dict[str, Any]:
    matches = match_features(kp1, des1, kp2, des2)
    result = {
        "matches": len(matches),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reconstructable": False,
        "reason": None,
    }

    if len(matches) < MIN_MATCHES:
        result["reason"] = f"Pas assez de correspondances ({len(matches)} < {MIN_MATCHES})."
        return result

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    h, w = image1.shape[:2]
    focal = max(h, w) * FOCAL_RATIO
    cx, cy = w / 2.0, h / 2.0

    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None or mask is None:
        result["reason"] = "Matrice essentielle non estimable."
        return result

    if E.shape[0] > 3:
        E = E[:3, :3]

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(np.sum(inlier_mask))
    result["inliers"] = inliers
    result["inlier_ratio"] = float(inliers / max(1, len(matches)))

    if inliers < MIN_INLIERS:
        result["reason"] = f"Pas assez d'inliers ({inliers} < {MIN_INLIERS})."
        return result

    points1 = pts1[inlier_mask]
    points2 = pts2[inlier_mask]

    try:
        _, R, t, pose_mask = cv2.recoverPose(E, points1, points2, K)
    except Exception as exc:
        result["reason"] = f"recoverPose failed: {exc}"
        return result

    pose_mask = pose_mask.ravel().astype(bool)
    points1 = points1[pose_mask]
    points2 = points2[pose_mask]

    if len(points1) < MIN_INLIERS:
        result["reason"] = "Trop peu de points après recoverPose."
        return result

    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))
    points4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points3d = (points4d[:3] / np.maximum(np.abs(points4d[3]), 1e-9)).T
    finite = np.all(np.isfinite(points3d), axis=1)
    points3d = points3d[finite]

    if len(points3d) > 10:
        distances = np.linalg.norm(points3d, axis=1)
        med = np.median(distances)
        mad = np.median(np.abs(distances - med))
        if mad > 1e-9:
            keep = np.abs(distances - med) < 8.0*mad
            points3d = points3d[keep]

    result.update({
        "reconstructable": len(points3d) >= MIN_INLIERS,
        "rotation": R.tolist(),
        "translation_direction": t.ravel().tolist(),
        "triangulated_points": int(len(points3d)),
        "points3d": points3d,
        "K": K,
    })

    if not result["reconstructable"]:
        result["reason"] = "Reconstruction 3D trop pauvre."

    return result


# Multi-view reconstruction
def reconstruct_scene(images: List[np.ndarray], view_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    features = []
    for index, image in enumerate(images):
        bbox = view_info[index]["loading_zone"]["bbox"]
        kp, des = create_features(image, bbox)
        features.append((kp, des))

    pair_results = []
    all_points = []

    for i in range(len(images)-1):
        for j in range(i+1, len(images)):
            result = estimate_pair_geometry(
                images[i], images[j],
                features[i][0], features[i][1],
                features[j][0], features[j][1],
            )

            public_result = {
                k: py(v) for k, v in result.items()
                if k not in ("points3d", "K", "rotation", "translation_direction")
            }
            pair_results.append({"view_a": i+1, "view_b": j+1, **public_result})

            if result.get("reconstructable"):
                points = result.get("points3d")
                if points is not None:
                    all_points.append(points)

    if not all_points:
        return {
            "success": False,
            "reason": "Aucune paire de vues n'a produit une reconstruction 3D suffisante.",
            "pair_results": pair_results,
            "point_count": 0,
            "volume_m3": None,
        }

    points = np.vstack(all_points)

    if len(points) > 20:
        centroid = np.median(points, axis=0)
        distances = np.linalg.norm(points-centroid, axis=1)
        q1, q3 = np.percentile(distances, [25,75])
        iqr = q3-q1
        if iqr > 0:
            keep = distances <= q3 + 2.5*iqr
            points = points[keep]

    return {
        "success": True,
        "pair_results": pair_results,
        "point_count": int(len(points)),
        "points3d": points,
        "scale_known": False,
        "scale_m": None,
        "volume_m3": None,
        "volume_method": None,
    }


# Metric scale
def estimate_scale_from_tires(views: List[Dict[str, Any]]) -> Dict[str, Any]:
    diameters = []
    for view in views:
        tire = view.get("tire")
        if tire:
            d = tire.get("diameter_px")
            if d and d > 0:
                diameters.append(float(d))

    if not diameters:
        return {
            "available": False,
            "reference_diameter_m": REFERENCE_TIRE_DIAMETER_M,
            "median_tire_diameter_px": None,
            "reason": "Aucun pneu exploitable détecté.",
        }

    return {
        "available": True,
        "reference_diameter_m": REFERENCE_TIRE_DIAMETER_M,
        "median_tire_diameter_px": float(np.median(diameters)),
        "reason": "Référence locale disponible.",
    }


# Point cloud volume
def point_cloud_volume(points: np.ndarray) -> Optional[float]:
    if not OPEN3D_AVAILABLE:
        return None
    if points is None or len(points) < 20:
        return None

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        if len(pcd.points) < 20:
            return None
        hull, _ = pcd.compute_convex_hull()
        volume = float(hull.get_volume())
        if not math.isfinite(volume):
            return None
        return volume
    except Exception as exc:
        print(f"[WARN] Point cloud hull failed: {exc}")
        return None


# Quality
def calculate_quality(views: List[Dict[str, Any]], reconstruction: Dict[str, Any]) -> Dict[str, Any]:
    truck_conf = [v["truck"]["confidence"] for v in views if v["truck"]["confidence"] is not None]
    load_conf = [v["wood"]["confidence"] for v in views]
    pairs = reconstruction.get("pair_results", [])

    if pairs:
        match_score = float(np.mean([min(1.0, p["matches"]/100.0) for p in pairs]))
        inlier_score = float(np.mean([p["inlier_ratio"] for p in pairs]))
        reconstruction_score = 0.45*match_score + 0.55*inlier_score
    else:
        reconstruction_score = 0.0

    truck_score = float(np.mean(truck_conf)) if truck_conf else 0.0
    load_score = float(np.mean(load_conf)) if load_conf else 0.0
    score = 0.20*truck_score + 0.20*load_score + 0.60*reconstruction_score

    return {
        "score": float(np.clip(score*100, 0, 100)),
        "truck_detection": truck_score,
        "wood_detection": load_score,
        "multi_view_geometry": reconstruction_score,
    }


# Endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "5.0.0",
        "yolo_loaded": yolo_model is not None,
        "open3d_loaded": OPEN3D_AVAILABLE,
    }


@app.post("/estimate-volume")
async def estimate_volume(photo1: UploadFile = File(...), photo2: UploadFile = File(...), photo3: UploadFile = File(...)):
    uploaded = [photo1, photo2, photo3]
    images = []

    for index, file in enumerate(uploaded, start=1):
        data = await file.read()
        image = decode_image(data)
        if image is None:
            raise HTTPException(status_code=400, detail=f"Photo {index} invalide.")
        images.append(image)

    print("\n" + "="*72)
    print("WOOD VOLUME ESTIMATOR V5")
    print("3 PHOTOS -> MULTI-VIEW GEOMETRY -> SPARSE 3D")
    print("="*72)

    view_info = []
    for index, image in enumerate(images, start=1):
        tire = detect_tire(image)
        truck = detect_truck(image)
        loading = detect_loading_zone(image, truck)
        wood = detect_wood(image, loading)

        info = {
            "photo": index,
            "resolution": [int(image.shape[1]), int(image.shape[0])],
            "tire": tire,
            "truck": {
                "detected": truck is not None,
                "confidence": float(truck["confidence"]) if truck else None,
                "bbox": truck["bbox"] if truck else None,
            },
            "loading_zone": loading,
            "wood": wood,
        }
        view_info.append(info)
        print(f"\n📷 PHOTO {index}")
        print("   Truck:", truck is not None)
        print("   Tire:", tire is not None)
        print("   Wood fill:", wood["fill_ratio"])

    reconstruction = reconstruct_scene(images, view_info)
    scale_info = estimate_scale_from_tires(view_info)
    sparse_volume = None

    if reconstruction.get("success"):
        sparse_volume = point_cloud_volume(reconstruction["points3d"])

    reconstruction_public = {k: v for k, v in reconstruction.items() if k != "points3d"}
    reconstruction_public["sparse_convex_hull_volume_unscaled"] = sparse_volume

    if sparse_volume is not None:
        reconstruction_public["volume_status"] = "unscaled_only"
        reconstruction_public["volume_warning"] = "Le volume du hull est dans l'échelle arbitraire de la reconstruction."
    else:
        reconstruction_public["volume_status"] = "not_available"

    quality = calculate_quality(view_info, reconstruction)

    warnings = [
        "Les trois images doivent représenter le même camion et le même chargement photographiés depuis des positions différentes.",
        "Un jeu de trois images générées séparément ne constitue pas un jeu photogrammétrique valide.",
        "Le pneu fournit une référence physique, mais V5 ne l'utilise pas comme facteur global sans correspondance 3D du pneu.",
        "Le nuage de points est encore sparse. Son convex hull éventuel n'est pas encore le volume du bois en m³.",
        "La segmentation du bois est heuristique.",
    ]

    response = {
        "success": True,
        "version": "5.0.0",
        "manual_dimensions": False,
        "dimensions_estimees_m": {
            "length_m": None,
            "width_m": None,
            "height_m": None,
            "status": "requires_dense_metric_3d",
        },
        "resultat": {
            "volume_apparent_m3": None,
            "stere_apparent": None,
            "volume_solid_indicative_m3": None,
            "status": "requires_metric_3d_scale_and_wood_surface",
        },
        "quality": quality,
        "scale": scale_info,
        "reconstruction_3d": reconstruction_public,
        "diagnostics": view_info,
        "warnings": warnings,
    }

    return py(response)


@app.post("/analyze")
async def analyze_compat(photo1: UploadFile = File(...), photo2: UploadFile = File(...), photo3: UploadFile = File(...)):
    return await estimate_volume(photo1, photo2, photo3)