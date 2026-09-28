"""
WOOD VOLUME ESTIMATOR V8
========================
3 photos -> multi-view geometry -> metric/geometry estimate.

V8 goals:
- Never multiply a 2D pixel fill ratio by a pseudo-volume.
- Never claim that a local tire pixel scale is a global metric scale.
- Attempt multi-view reconstruction with SIFT + F/E matrices.
- Estimate a conservative load envelope from consistent projections.
- Use tire only as a physical reference for diagnostics.
- Return a volume only when a geometry path passes validation.
- If validation fails, return status="not_reliably_computable" rather than a fabricated m3.

This is still a prototype. Production accuracy requires calibrated camera
intrinsics, real photos of the same scene, robust tire/wood correspondences,
and preferably COLMAP/Open3D or an equivalent dense MVS pipeline.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

VERSION = "8.0.0"

YOLO_MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n-seg.pt")
REFERENCE_TIRE_DIAMETER_M = float(os.getenv("REFERENCE_TIRE_DIAMETER_M", "1.00"))

# Matching / geometry
SIFT_FEATURES = int(os.getenv("SIFT_FEATURES", "10000"))
LOWE_RATIO = float(os.getenv("LOWE_RATIO", "0.78"))
MIN_F_INLIERS = int(os.getenv("MIN_F_INLIERS", "12"))
MIN_E_INLIERS = int(os.getenv("MIN_E_INLIERS", "10"))
MIN_TRIANGULATED = int(os.getenv("MIN_TRIANGULATED", "20"))
FOCAL_RATIO = float(os.getenv("FOCAL_RATIO", "1.20"))

# Conservative acceptance thresholds
MIN_INLIER_RATIO = float(os.getenv("MIN_INLIER_RATIO", "0.12"))
MIN_PAIR_COUNT = int(os.getenv("MIN_PAIR_COUNT", "2"))

OPEN3D_AVAILABLE = False
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception:
    o3d = None


app = FastAPI(title="Wood Volume Estimator V8", version=VERSION)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/tire.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    yolo_model = YOLO(YOLO_MODEL_NAME)
except Exception as exc:
    yolo_model = None
    print(f"[WARN] YOLO unavailable: {exc}")


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


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def decode_image(data: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def detect_tire(image: np.ndarray) -> Optional[Dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    h, w = gray.shape

    min_r = max(12, int(min(h, w) * 0.025))
    max_r = max(min_r + 5, int(min(h, w) * 0.20))

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(25, int(min(h, w) * 0.07)),
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
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(w, x+r), min(h, y+r)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        mean = float(np.mean(roi))
        # Circularity/edge evidence around the candidate.
        edges = cv2.Canny(roi, 60, 140)
        edge_density = float(np.mean(edges > 0))
        score = 2.0 * r + max(0.0, 120.0 - mean) + 250.0 * edge_density
        candidates.append((score, x, y, r))

    if not candidates:
        return None

    _, x, y, r = max(candidates, key=lambda z: z[0])
    return {
        "cx_px": float(x),
        "cy_px": float(y),
        "radius_px": float(r),
        "diameter_px": float(2 * r),
        "reference_diameter_m": REFERENCE_TIRE_DIAMETER_M,
    }


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
            cid = int(box.cls[0])
            name = str(names.get(cid, cid)).lower()
            if name != "truck":
                continue
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(float)
            area = max(1.0, (bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
            cand = (conf, area, result, i, bbox)
            if best is None or cand[0] > best[0] or (
                cand[0] == best[0] and cand[1] > best[1]
            ):
                best = cand

    if best is None:
        return None

    conf, _, result, index, bbox = best
    mask = None
    if getattr(result, "masks", None) is not None and result.masks.data is not None:
        try:
            mask = result.masks.data[index].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        except Exception:
            mask = None

    return {
        "confidence": conf,
        "bbox": [float(x) for x in bbox],
        "mask_available": mask is not None,
        "mask": mask,
    }


def loading_zone(image: np.ndarray, truck: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if truck is None:
        return {"bbox": None, "confidence": 0.0}

    h, w = image.shape[:2]
    x1, y1, x2, y2 = truck["bbox"]
    bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)

    # Deliberately broad. This is a proposal, not a physical measurement.
    lx1 = int(max(0, x1 + 0.12*bw))
    lx2 = int(min(w, x2 - 0.02*bw))
    ly1 = int(max(0, y1 + 0.05*bh))
    ly2 = int(min(h, y1 + 0.82*bh))

    return {
        "bbox": [lx1, ly1, max(lx1+1, lx2), max(ly1+1, ly2)],
        "confidence": float(np.clip(0.45 + 0.45*truck["confidence"], 0, 0.95)),
    }


def wood_diagnostics(image: np.ndarray, zone: Dict[str, Any]) -> Dict[str, Any]:
    bbox = zone.get("bbox")
    if bbox is None:
        return {"pixel_coverage": None, "confidence": 0.0, "method": "unavailable"}

    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"pixel_coverage": None, "confidence": 0.0, "method": "empty"}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (9, 9), 0)
    threshold = max(5.0, float(np.percentile(texture, 55)))
    candidate = (
        (texture >= threshold)
        & (hsv[:, :, 2] > 30)
        & (hsv[:, :, 2] < 250)
        & (hsv[:, :, 1] > 15)
    )
    mask = cv2.morphologyEx(
        (candidate.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
    )
    coverage = float(np.mean(mask > 0))
    return {
        "pixel_coverage": coverage,
        "confidence": float(np.clip(0.25 + 0.55*min(1.0, coverage*2), 0, 0.80)),
        "method": "heuristic_texture_segmentation",
    }


def feature_roi(image: np.ndarray, truck: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    if truck is None:
        return None
    x1, y1, x2, y2 = truck["bbox"]
    h, w = image.shape[:2]
    # Expand around truck to include stable structural features.
    pad_x = 0.08 * (x2-x1)
    pad_y = 0.08 * (y2-y1)
    return [
        int(max(0, x1-pad_x)),
        int(max(0, y1-pad_y)),
        int(min(w, x2+pad_x)),
        int(min(h, y2+pad_y)),
    ]


def create_features(image: np.ndarray, roi: Optional[List[int]]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = None
    if roi:
        x1, y1, x2, y2 = roi
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=0.015,
        edgeThreshold=12,
        sigma=1.6,
    )
    return sift.detectAndCompute(gray, mask)


def match_features(kp1, des1, kp2, des2) -> List[cv2.DMatch]:
    if des1 is None or des2 is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO*n.distance:
            good.append(m)
    return good


def pair_geometry(image1, image2, kp1, des1, kp2, des2) -> Dict[str, Any]:
    matches = match_features(kp1, des1, kp2, des2)
    out = {
        "matches": len(matches),
        "fundamental_inliers": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reconstructable": False,
        "reason": None,
        "triangulated_points": 0,
        "points3d": None,
    }
    if len(matches) < 12:
        out["reason"] = f"Pas assez de matches ({len(matches)} < 12)."
        return out

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, fmask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, 1.5, 0.999
    )
    if F is None or fmask is None:
        out["reason"] = "Matrice fondamentale non estimable."
        return out

    if F.shape[0] > 3:
        F = F[:3, :3]

    fm = fmask.ravel().astype(bool)
    finliers = int(np.sum(fm))
    out["fundamental_inliers"] = finliers

    if finliers < MIN_F_INLIERS:
        out["reason"] = f"Pas assez d'inliers F ({finliers} < {MIN_F_INLIERS})."
        return out

    p1 = pts1[fm]
    p2 = pts2[fm]
    h, w = image1.shape[:2]
    focal = max(h, w) * FOCAL_RATIO
    K = np.array(
        [[focal, 0, w/2.0], [0, focal, h/2.0], [0, 0, 1]],
        dtype=np.float64,
    )

    E, emask = cv2.findEssentialMat(
        p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5
    )
    if E is None or emask is None:
        out["reason"] = "Matrice essentielle non estimable."
        return out

    if E.shape[0] > 3:
        E = E[:3, :3]

    em = emask.ravel().astype(bool)
    einliers = int(np.sum(em))
    out["inliers"] = einliers
    out["inlier_ratio"] = float(einliers / max(1, len(matches)))

    if einliers < MIN_E_INLIERS or out["inlier_ratio"] < MIN_INLIER_RATIO:
        out["reason"] = (
            f"Géométrie insuffisante ({einliers} inliers, "
            f"ratio={out['inlier_ratio']:.3f})."
        )
        return out

    p1 = p1[em]
    p2 = p2[em]

    try:
        _, R, t, pose_mask = cv2.recoverPose(E, p1, p2, K)
    except Exception as exc:
        out["reason"] = f"recoverPose failed: {exc}"
        return out

    pm = pose_mask.ravel().astype(bool)
    p1 = p1[pm]
    p2 = p2[pm]

    if len(p1) < MIN_TRIANGULATED:
        out["reason"] = f"Trop peu de points après recoverPose ({len(p1)})."
        return out

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    X4 = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    X = (X4[:3] / np.maximum(np.abs(X4[3]), 1e-9)).T

    finite = np.all(np.isfinite(X), axis=1)
    X = X[finite]
    if len(X) < MIN_TRIANGULATED:
        out["reason"] = "Nuage 3D trop pauvre après triangulation."
        return out

    # Robust distance filtering.
    d = np.linalg.norm(X, axis=1)
    med = np.median(d)
    mad = np.median(np.abs(d-med))
    if mad > 1e-9:
        X = X[np.abs(d-med) < 8.0*mad]

    out["triangulated_points"] = int(len(X))
    out["points3d"] = X
    out["reconstructable"] = len(X) >= MIN_TRIANGULATED
    out["reason"] = None if out["reconstructable"] else "Nuage 3D insuffisant."
    return out


def reconstruct(images: List[np.ndarray], views: List[Dict[str, Any]]) -> Dict[str, Any]:
    features = []
    for i, image in enumerate(images):
        roi = feature_roi(image, views[i].get("truck_raw"))
        kp, des = create_features(image, roi)
        features.append((kp, des))

    pairs = []
    all_points = []
    for i in range(2):
        for j in range(i+1, 3):
            r = pair_geometry(
                images[i], images[j],
                features[i][0], features[i][1],
                features[j][0], features[j][1],
            )
            public = {k: py(v) for k, v in r.items() if k != "points3d"}
            pairs.append({"view_a": i+1, "view_b": j+1, **public})
            if r.get("reconstructable") and r.get("points3d") is not None:
                all_points.append(r["points3d"])

    if not all_points:
        return {
            "success": False,
            "reason": "Aucune paire n'a produit une reconstruction 3D suffisamment robuste.",
            "pair_results": pairs,
            "point_count": 0,
            "scale_known": False,
            "metric_volume_m3": None,
        }

    X = np.vstack(all_points)
    if len(X) > 30:
        c = np.median(X, axis=0)
        d = np.linalg.norm(X-c, axis=1)
        q1, q3 = np.percentile(d, [25, 75])
        iqr = q3-q1
        if iqr > 0:
            X = X[d <= q3 + 2.5*iqr]

    return {
        "success": True,
        "reason": None,
        "pair_results": pairs,
        "point_count": int(len(X)),
        "scale_known": False,
        "metric_volume_m3": None,
    }


def estimate_quality(views, reconstruction):
    truck = [v["truck"]["confidence"] for v in views if v["truck"]["confidence"] is not None]
    wood = [v["wood"]["confidence"] for v in views]
    pairs = reconstruction.get("pair_results", [])

    if pairs:
        geometry = float(np.mean([
            min(1.0, p["matches"]/100.0) * 0.35 +
            p["inlier_ratio"] * 0.65
            for p in pairs
        ]))
    else:
        geometry = 0.0

    t = float(np.mean(truck)) if truck else 0.0
    w = float(np.mean(wood)) if wood else 0.0
    score = 100.0 * (0.20*t + 0.20*w + 0.60*geometry)

    return {
        "score": float(np.clip(score, 0, 100)),
        "truck_detection": t,
        "wood_detection": w,
        "multi_view_geometry": geometry,
    }


def projected_box_diagnostic(views) -> Dict[str, Any]:
    """
    V8 deliberately does NOT turn this into m3.

    It reports the apparent pixel envelope and local tire scales so that
    calibration/future metric reconstruction can be inspected.
    """
    samples = []
    for v in views:
        tire = v["tire"]
        zone = v["loading_zone"]["bbox"]
        if not tire or not zone:
            continue
        x1, y1, x2, y2 = zone
        d = tire["diameter_px"]
        samples.append({
            "photo": v["photo"],
            "load_width_px": float(x2-x1),
            "load_height_px": float(y2-y1),
            "tire_diameter_px": float(d),
            "local_scale_m_per_px": REFERENCE_TIRE_DIAMETER_M / d,
            "pixel_coverage": v["wood"]["pixel_coverage"],
        })

    return {
        "available": len(samples) == 3,
        "samples": samples,
        "used_for_volume": False,
        "reason": (
            "Les projections 2D et l'échelle locale du pneu sont conservées "
            "comme diagnostics; elles ne sont pas transformées directement en m³."
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "yolo_loaded": yolo_model is not None,
        "open3d_loaded": OPEN3D_AVAILABLE,
        "manual_dimensions": False,
        "metric_volume_requires_validated_3d": True,
    }


@app.post("/estimate-volume")
async def estimate_volume(
    photo1: UploadFile = File(...),
    photo2: UploadFile = File(...),
    photo3: UploadFile = File(...),
):
    files = [photo1, photo2, photo3]
    images = []

    for i, f in enumerate(files, start=1):
        data = await f.read()
        image = decode_image(data)
        if image is None:
            raise HTTPException(status_code=400, detail=f"Photo {i} invalide.")
        images.append(image)

    print("\n" + "="*72)
    print("WOOD VOLUME ESTIMATOR V8")
    print("3 PHOTOS -> 3D RECONSTRUCTION -> VALIDATED METRIC OUTPUT")
    print("="*72)

    views = []
    for i, image in enumerate(images, start=1):
        tire = detect_tire(image)
        truck = detect_truck(image)
        zone = loading_zone(image, truck)
        wood = wood_diagnostics(image, zone)

        view = {
            "photo": i,
            "resolution": [int(image.shape[1]), int(image.shape[0])],
            "tire": tire,
            "truck": {
                "detected": truck is not None,
                "confidence": float(truck["confidence"]) if truck else None,
                "bbox": truck["bbox"] if truck else None,
            },
            "truck_raw": truck,
            "loading_zone": zone,
            "wood": wood,
        }
        views.append(view)

        print(
            f"PHOTO {i}: truck={truck is not None}, "
            f"tire={tire is not None}, wood={wood['pixel_coverage']}"
        )

    reconstruction = reconstruct(images, views)
    quality = estimate_quality(views, reconstruction)
    projection_diag = projected_box_diagnostic(views)

    # V8 does not claim a volume unless a validated metric 3D reconstruction exists.
    metric_volume = None
    dimensions = {
        "length_m": None,
        "width_m": None,
        "height_m": None,
        "status": "requires_validated_metric_3d",
    }

    warnings = [
        "Les trois photos doivent être des prises de vue réelles du même camion et du même chargement.",
        "Les images générées séparément par IA ne constituent pas un jeu photogrammétrique métrique fiable.",
        "La couverture de pixels du bois est un diagnostic 2D et n'est pas utilisée comme facteur de volume.",
        "Le diamètre du pneu est une référence physique locale; il n'est pas utilisé comme facteur m/px global.",
        "V8 ne fabrique pas un volume m³ lorsque la reconstruction 3D métrique n'est pas validée.",
    ]

    if reconstruction["success"]:
        status = "reconstruction_sparse_non_metric"
        warnings.append(
            "Une reconstruction sparse existe, mais son échelle métrique et sa surface bois ne sont pas encore validées."
        )
    else:
        status = "not_reliably_computable"
        warnings.append(
            "La reconstruction multi-vues n'a pas atteint les critères de robustesse."
        )

    diagnostics_public = []
    for v in views:
        diagnostics_public.append({
            k: py(val)
            for k, val in v.items()
            if k != "truck_raw"
        })

    response = {
        "success": True,
        "version": VERSION,
        "manual_dimensions": False,
        "dimensions_estimees_m": dimensions,
        "resultat": {
            "volume_apparent_m3": metric_volume,
            "stere_apparent": None,
            "volume_solid_indicative_m3": None,
            "method": "validated_metric_3d_only",
            "status": status,
        },
        "quality": quality,
        "scale": {
            "reference_diameter_m": REFERENCE_TIRE_DIAMETER_M,
            "detected_in_all_views": all(v["tire"] is not None for v in views),
            "used_for_global_scale": False,
            "reason": (
                "Le pneu n'est utilisé comme échelle globale qu'après "
                "correspondance 3D validée."
            ),
        },
        "projection_diagnostics": projection_diag,
        "reconstruction_3d": reconstruction,
        "diagnostics": diagnostics_public,
        "methodology": {
            "detection": "YOLO truck detection.",
            "wood": "2D heuristic diagnostic only; not used as volumetric fill factor.",
            "features": "SIFT + Lowe ratio test.",
            "camera_geometry": "Fundamental Matrix + Essential Matrix + recoverPose.",
            "triangulation": "Sparse 3D triangulation with robust filtering.",
            "metric_scale": "Not declared until a valid 3D metric reference is established.",
            "volume": "Declared in m3 only after validated metric 3D + wood surface.",
        },
        "warnings": warnings,
    }

    return py(response)


@app.post("/analyze")
async def analyze_compat(
    photo1: UploadFile = File(...),
    photo2: UploadFile = File(...),
    photo3: UploadFile = File(...),
):
    return await estimate_volume(photo1, photo2, photo3)
