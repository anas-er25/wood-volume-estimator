# Wood Volume Estimator V8

V8 removes the invalid V7 calculation:

`projected_box_volume × heuristic_pixel_fill_ratio`

That pixel coverage is now diagnostic only.

## Run

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend_tire_reference_v8:app --reload
```

Open:

http://127.0.0.1:8000/

## Important

V8 does NOT fabricate a metric volume when metric 3D reconstruction is not validated.

For a real metric volume, use three real photographs of the same truck/load at the same moment, with sufficient overlap and camera movement. The next production step is calibrated intrinsics + COLMAP/Open3D dense MVS + trained wood-load segmentation + validated metric scale.
