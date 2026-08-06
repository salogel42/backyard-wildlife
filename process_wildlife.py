import numpy as np
import os
import csv
import shutil
from datetime import datetime
from PIL import Image
import cv2
from PytorchWildlife.models import detection as pw_detection
import supervision as sv

# 1. Paths configuration
BASE_ARCHIVE_DIR = os.path.expanduser("~/BackyardWildlifeArchive")
INCOMING_DIR = os.path.join(BASE_ARCHIVE_DIR, "incoming")
ANNOTATED_DIR = os.path.join(BASE_ARCHIVE_DIR, "output/annotated")
HIGH_CONF_DIR = os.path.join(BASE_ARCHIVE_DIR, "output/high_confidence")
MED_CONF_DIR = os.path.join(BASE_ARCHIVE_DIR, "output/med_confidence")
LOW_CONF_DIR = os.path.join(BASE_ARCHIVE_DIR, "output/low_confidence")
EMPTIES_DIR = os.path.join(BASE_ARCHIVE_DIR, "output/empties")
# Static/background false positives filtered by repeat-detection elimination.
FALSE_POS_DIR = os.path.join(BASE_ARCHIVE_DIR, "output/false_positives")
OUTPUT_CSV = "backyard_wildlife_log.csv"

# Ensure destination directories exist
os.makedirs(HIGH_CONF_DIR, exist_ok=True)
os.makedirs(MED_CONF_DIR, exist_ok=True)
os.makedirs(LOW_CONF_DIR, exist_ok=True)
os.makedirs(EMPTIES_DIR, exist_ok=True)
os.makedirs(FALSE_POS_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# --- Tunable behavior (overridable via environment variables) ---------------

# Which MegaDetector v6 model to load. Larger variants (e.g. MDV6-yolov9-e) are
# slower but sometimes more accurate; note a bigger model does NOT reliably fix
# static-background false positives -- repeat-detection elimination below does.
MODEL_VERSION = os.environ.get("WILDLIFE_MODEL_VERSION", "MDV6-yolov9-c")

# MegaDetector's single_image_detection() defaults to an internal cutoff of 0.2
# and silently drops everything below it. That turns faint / distant / partial
# animals into "empties" before the tiered routing below ever sees them. We pass
# a lower threshold so those low-confidence hits reach the low_confidence tier
# instead of disappearing. Raise this toward 0.2 if you get too many false hits.
DETECTION_CONF_THRESHOLD = float(os.environ.get("WILDLIFE_CONF_THRESHOLD", "0.1"))

# Repeat-detection elimination (RDE). A fixed trail cam repeatedly flags the same
# static object -- a branch, trunk, or rock -- as an "animal" in nearly the same
# spot across many photos. RDE clusters animal boxes by location across all the
# photos from one camera in this run; any location that recurs in at least
# RDE_MIN_REPEATS distinct photos is treated as background and its boxes are
# removed. A real animal that merely walks through that spot once is kept,
# because it does not recur in the same box across many frames.
RDE_ENABLED = os.environ.get("WILDLIFE_RDE", "1") != "0"
RDE_MIN_REPEATS = int(os.environ.get("WILDLIFE_RDE_MIN_REPEATS", "10"))
# Safety net for animals that use a spot which also produces false positives
# (e.g. a squirrel on the same oak trunk that gets flagged as background). A
# detection at or above this confidence is never removed by RDE, so a strong
# detection is kept for review even if it sits on a recurring background box.
# The static tree boxes here top out well below this, so they still get filtered.
RDE_KEEP_CONF = float(os.environ.get("WILDLIFE_RDE_KEEP_CONF", "0.7"))
# How much two boxes must overlap to count as "the same location". Tuned so it
# absorbs the frame-to-frame wobble of a static object's box (the detector emits
# slightly different boxes for the same branch as light changes) while staying
# above the overlap a one-off real animal has with that region. Combined with
# the RDE_MIN_REPEATS frequency test, this keeps a real animal that walks
# through the spot once (it isn't there in >= RDE_MIN_REPEATS frames).
RDE_IOU_THRESHOLD = float(os.environ.get("WILDLIFE_RDE_IOU", "0.6"))

print(f"Loading MegaDetector v6 model ({MODEL_VERSION})...")
detection_model = pw_detection.MegaDetectorV6(version=MODEL_VERSION)

ANIMAL_CLASS_ID = 0  # MegaDetector: 0=animal, 1=person, 2=vehicle
results_data = []
photo_extensions = (".jpg", ".jpeg", ".png")
video_extensions = (".mp4", ".avi", ".mov")


def annotate_image_with_detections(img, detections):
    """Draws bounding boxes and labels on a PIL Image using Supervision."""
    img_np = np.array(img)

    if detections is not None and len(detections) > 0:
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [
            f"animal {conf:.2f}" if cid == ANIMAL_CLASS_ID else f"class_{cid} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = box_annotator.annotate(scene=img_np.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return Image.fromarray(annotated_frame)

    return img


def get_file_timestamp(file_path):
    """Extracts timestamp and adds microsecond uniqueness to prevent 000000 collisions."""
    try:
        mod_time = os.path.getmtime(file_path)
        dt_obj = datetime.fromtimestamp(mod_time)
        if dt_obj.microsecond == 0:
            dt_obj = dt_obj.replace(microsecond=datetime.now().microsecond)
        return dt_obj, dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        dt_obj = datetime.now()
        return dt_obj, dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def max_animal_conf(detections):
    """Highest 'animal' confidence in a Detections object (0.0 if none)."""
    if detections is None or len(detections) == 0:
        return 0.0
    confs = [
        float(conf)
        for cid, conf in zip(detections.class_id, detections.confidence)
        if cid == ANIMAL_CLASS_ID and conf > 0.0
    ]
    return max(confs) if confs else 0.0


def detect_image(image_path):
    """Runs MegaDetector on a single image.

    Returns (detections, normalized_coords) where detections is an
    sv.Detections and normalized_coords is a list of [x1, y1, x2, y2] boxes
    normalized to [0, 1]. Returns (None, []) on error.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        result = detection_model.single_image_detection(img_array, det_conf_thres=DETECTION_CONF_THRESHOLD)
        return result.get("detections"), result.get("normalized_coords", [])
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, []


def process_video(video_path):
    """Samples frames from a video file and runs MegaDetector.

    Returns (max_confidence, annotated_img). Videos are not run through
    repeat-detection elimination (moving scenes rarely produce the same static
    false positive), so their result is finalized here.
    """
    max_conf = 0.0
    annotated_img = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 0.0, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Fallback assumption

    step = int(fps)
    if step < 1:
        step = 1

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            result = detection_model.single_image_detection(np.array(pil_img), det_conf_thres=DETECTION_CONF_THRESHOLD)
            detections = result.get("detections")
            frame_max_conf = max_animal_conf(detections)

            if frame_max_conf > max_conf:
                max_conf = frame_max_conf
                annotated_img = annotate_image_with_detections(pil_img, detections)

        current_frame += 1

    cap.release()
    return max_conf, annotated_img


def _iou(a, b):
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# Cap on representative boxes kept per cluster, so single-linkage clustering
# stays bounded (O(files * clusters * MAX_CLUSTER_REPS)) on large batches.
MAX_CLUSTER_REPS = 32


def find_static_locations(photo_items):
    """Cluster animal boxes across a camera's photos and return the box
    locations that recur in at least RDE_MIN_REPEATS distinct photos.

    Uses single-linkage clustering: a box joins a cluster if it overlaps ANY of
    the cluster's representative boxes (not just the first one). This matters
    because the detector emits slightly different box extents for the same static
    object across frames; single-rep clustering fragments those variants into
    several small clusters that each miss the repeat threshold, whereas
    single-linkage chains them into one cluster that recurs often enough to be
    recognized as background.

    photo_items: list of dicts with keys 'detections' (sv.Detections) and
    'normalized_coords' (list aligned with detections).
    Returns a flat list of representative [x1, y1, x2, y2] normalized boxes from
    every cluster deemed static (pass 2 filters a detection if it overlaps any).
    """
    if not RDE_ENABLED:
        return []

    clusters = []  # each: {"reps": [box, ...], "files": set()}
    for idx, item in enumerate(photo_items):
        detections = item["detections"]
        norm = item["normalized_coords"]
        if detections is None or len(detections) == 0:
            continue
        for i, box in enumerate(norm):
            if int(detections.class_id[i]) != ANIMAL_CLASS_ID:
                continue
            box = list(box)
            hits = [c for c in clusters if any(_iou(r, box) >= RDE_IOU_THRESHOLD for r in c["reps"])]
            if hits:
                base = hits[0]
                for other in hits[1:]:  # a box can bridge several clusters -> merge
                    base["files"] |= other["files"]
                    base["reps"] += other["reps"]
                    clusters.remove(other)
                base["files"].add(idx)
                # Keep a novel box as a new representative, bounded in count.
                if len(base["reps"]) < MAX_CLUSTER_REPS and all(_iou(r, box) < 0.95 for r in base["reps"]):
                    base["reps"].append(box)
                base["reps"] = base["reps"][:MAX_CLUSTER_REPS]
            else:
                clusters.append({"reps": [box], "files": {idx}})

    static = []
    for c in clusters:
        if len(c["files"]) >= RDE_MIN_REPEATS:
            static.extend(c["reps"])
    return static


def unique_dest_name(base_slug, ext):
    """Return a destination base name whose file (base + ext) does not already
    exist in any archive tier, so two captures that share a timestamp can't
    overwrite each other. The extension must be included in the check because
    that is how the files are actually stored.
    """
    counter = 1
    final = base_slug
    while any(
        os.path.exists(os.path.join(d, final + ext))
        for d in (HIGH_CONF_DIR, MED_CONF_DIR, LOW_CONF_DIR, EMPTIES_DIR, FALSE_POS_DIR)
    ):
        final = f"{base_slug}_{counter}"
        counter += 1
    return final


# Scan incoming subfolders (e.g., cam1, cam2)
if not os.path.exists(INCOMING_DIR):
    print(f"Incoming directory not found: {INCOMING_DIR}")
    exit(1)

subfolders = [d for d in os.listdir(INCOMING_DIR) if os.path.isdir(os.path.join(INCOMING_DIR, d))]
if not subfolders:
    print(f"No camera subfolders found in {INCOMING_DIR}. Create cam1, cam2, etc., and drop files there!")
    exit(0)

print(f"Found camera staging folders: {subfolders}")

for camera_name in subfolders:
    cam_folder_path = os.path.join(INCOMING_DIR, camera_name)
    files = sorted(os.listdir(cam_folder_path))

    if not files:
        print(f"Skipping {camera_name} (empty folder).")
        continue

    print(f"\n--- Processing {camera_name} ({len(files)} items) ---")

    # Pass 1: run detection on every file. For photos we store only lightweight
    # detection metadata (no images) so this scales to thousands of files.
    photo_items = []
    video_items = []
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in photo_extensions and ext not in video_extensions:
            continue

        src_path = os.path.join(cam_folder_path, filename)
        dt_obj, timestamp_str = get_file_timestamp(src_path)

        if ext in photo_extensions:
            detections, norm = detect_image(src_path)
            photo_items.append({
                "filename": filename, "src_path": src_path, "ext": ext,
                "dt_obj": dt_obj, "timestamp_str": timestamp_str,
                "detections": detections, "normalized_coords": norm,
            })
        else:
            max_conf, annotated_img = process_video(src_path)
            video_items.append({
                "filename": filename, "src_path": src_path, "ext": ext,
                "dt_obj": dt_obj, "timestamp_str": timestamp_str,
                "max_conf": max_conf, "annotated_img": annotated_img,
            })

    # Identify static/background locations for this camera.
    static_locations = find_static_locations(photo_items)
    if static_locations:
        print(f"  [RDE] {len(static_locations)} recurring background location(s) "
              f"detected in >= {RDE_MIN_REPEATS} photos; filtering them out.")

    # Pass 2: filter static detections, route, annotate survivors, move, log.
    for item in photo_items:
        detections = item["detections"]
        filename = item["filename"]
        timestamp_str = item["timestamp_str"]
        print(f"\n\n**Processing photo:** {filename}")

        had_animal = max_animal_conf(detections) > 0.0

        # Build a keep-mask that drops animal boxes matching a static location.
        kept = detections
        if detections is not None and len(detections) > 0 and static_locations:
            norm = item["normalized_coords"]
            mask = np.ones(len(detections), dtype=bool)
            for i in range(len(detections)):
                if int(detections.class_id[i]) != ANIMAL_CLASS_ID:
                    continue
                if float(detections.confidence[i]) >= RDE_KEEP_CONF:
                    continue  # trust strong detections (e.g. a squirrel on the oak)
                if any(_iou(loc, norm[i]) >= RDE_IOU_THRESHOLD for loc in static_locations):
                    mask[i] = False
            kept = detections[mask]

        max_conf = max_animal_conf(kept)
        filtered_by_rde = had_animal and max_conf == 0.0

        dt_obj = item["dt_obj"]
        conf_int = int(round(max_conf, 2) * 100)
        conf_prefix = f"conf{conf_int:03d}"
        time_slug = dt_obj.strftime("%Y-%m-%d_%H-%M-%S-%f")
        base_slug = f"{conf_prefix}_{camera_name}_{time_slug}"
        final_dest_name = unique_dest_name(base_slug, item["ext"])
        new_filename = f"{final_dest_name}{item['ext']}"

        # Route into final archive based on tiered confidence.
        if filtered_by_rde:
            print(f"  -> [FALSE POSITIVE / RDE] [{timestamp_str}] recurring background box removed -> false_positives/")
            dest_path = os.path.join(FALSE_POS_DIR, new_filename)
            status_str = "Repeat Detection (filtered)"
        elif max_conf >= 0.7:
            print(f"  -> [ANIMAL FOUND] [{timestamp_str}] (Conf: {max_conf:.2f}) -> Saved to high_confidence/")
            dest_path = os.path.join(HIGH_CONF_DIR, new_filename)
            status_str = "Animal Detected"
        elif max_conf >= 0.3:
            print(f"  -> [ANIMAL MAYBE FOUND] [{timestamp_str}] (Conf: {max_conf:.2f}) -> Saved to med_confidence/")
            dest_path = os.path.join(MED_CONF_DIR, new_filename)
            status_str = "Animal Detected"
        elif max_conf > 0:
            print(f"  -> [LOW CONFIDENCE] [{timestamp_str}] (Conf: {max_conf:.2f}) -> Saved to low_confidence/")
            dest_path = os.path.join(LOW_CONF_DIR, new_filename)
            status_str = "Low Confidence"
        else:
            print(f"  -> [Empty] Archiving to empties/")
            dest_path = os.path.join(EMPTIES_DIR, new_filename)
            status_str = "Empty"

        # Annotate surviving detections (re-open the image only when needed).
        annotated_image_path = None
        if max_conf > 0 and kept is not None and len(kept) > 0:
            try:
                img = Image.open(item["src_path"]).convert("RGB")
                annotated_img = annotate_image_with_detections(img, kept)
                annotated_image_path = os.path.join(ANNOTATED_DIR, f"{final_dest_name}.jpg")
                annotated_img.save(annotated_image_path)
            except Exception as e:
                print(f"  (could not annotate {filename}: {e})")

        shutil.move(item["src_path"], dest_path)

        current_result = {
            "timestamp": timestamp_str,
            "camera": camera_name,
            "original_filename": filename,
            "archived_filename": new_filename,
            "annotated_filename": annotated_image_path,
            "status": status_str,
            "max_confidence": round(float(max_conf), 2),
        }
        print(current_result)

        # Log everything except pure empties (RDE-filtered files are logged so
        # you can audit what was removed as background).
        if max_conf > 0 or filtered_by_rde:
            results_data.append(current_result)

    # Videos: no RDE, route by their aggregated confidence.
    for item in video_items:
        filename = item["filename"]
        timestamp_str = item["timestamp_str"]
        max_conf = item["max_conf"]
        annotated_img = item["annotated_img"]
        print(f"\n\n**Processing video:** {filename}")

        dt_obj = item["dt_obj"]
        conf_int = int(round(max_conf, 2) * 100)
        conf_prefix = f"conf{conf_int:03d}"
        time_slug = dt_obj.strftime("%Y-%m-%d_%H-%M-%S-%f")
        base_slug = f"{conf_prefix}_{camera_name}_{time_slug}"
        final_dest_name = unique_dest_name(base_slug, item["ext"])
        new_filename = f"{final_dest_name}{item['ext']}"

        if max_conf >= 0.7:
            print(f"  -> [ANIMAL FOUND] [{timestamp_str}] (Conf: {max_conf:.2f}) -> Saved to high_confidence/")
            dest_path = os.path.join(HIGH_CONF_DIR, new_filename)
            status_str = "Animal Detected"
        elif max_conf >= 0.3:
            print(f"  -> [ANIMAL MAYBE FOUND] [{timestamp_str}] (Conf: {max_conf:.2f}) -> Saved to med_confidence/")
            dest_path = os.path.join(MED_CONF_DIR, new_filename)
            status_str = "Animal Detected"
        elif max_conf > 0:
            print(f"  -> [LOW CONFIDENCE] [{timestamp_str}] (Conf: {max_conf:.2f}) -> Saved to low_confidence/")
            dest_path = os.path.join(LOW_CONF_DIR, new_filename)
            status_str = "Low Confidence"
        else:
            print(f"  -> [Empty] Archiving to empties/")
            dest_path = os.path.join(EMPTIES_DIR, new_filename)
            status_str = "Empty"

        annotated_image_path = None
        if max_conf > 0 and annotated_img is not None:
            annotated_image_path = os.path.join(ANNOTATED_DIR, f"{final_dest_name}.jpg")
            annotated_img.save(annotated_image_path)

        shutil.move(item["src_path"], dest_path)

        current_result = {
            "timestamp": timestamp_str,
            "camera": camera_name,
            "original_filename": filename,
            "archived_filename": new_filename,
            "annotated_filename": annotated_image_path,
            "status": status_str,
            "max_confidence": round(float(max_conf), 2),
        }
        print(current_result)

        if max_conf > 0:
            results_data.append(current_result)

# Save results to CSV log
csv_path = os.path.join(BASE_ARCHIVE_DIR, OUTPUT_CSV)
if results_data:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as csv_file:
        fieldnames = ["timestamp", "camera", "original_filename", "archived_filename", "annotated_filename", "status", "max_confidence"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in results_data:
            writer.writerow(row)
    print(f"\nDone! Logged {len(results_data)} detection hits to {csv_path}")
else:
    print("\nDone! No detections found across scanned folders.")

print("All staged files processed and renamed successfully.")
