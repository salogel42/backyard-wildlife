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
OUTPUT_CSV = "backyard_wildlife_log.csv"

# MegaDetector's single_image_detection() defaults to an internal cutoff of 0.2
# and silently drops everything below it. That turns faint / distant / partial
# animals into "empties" before the tiered routing below ever sees them. We pass
# a lower threshold so those low-confidence hits reach the low_confidence tier
# instead of disappearing. Raise this toward 0.2 if you get too many false hits.
DETECTION_CONF_THRESHOLD = 0.1

# Ensure destination directories exist
os.makedirs(HIGH_CONF_DIR, exist_ok=True)
os.makedirs(MED_CONF_DIR, exist_ok=True)
os.makedirs(LOW_CONF_DIR, exist_ok=True)
os.makedirs(EMPTIES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

print("Loading MegaDetector v6 model...")
detection_model = pw_detection.MegaDetectorV6(version="MDV6-yolov9-c")

results_data = []
photo_extensions = (".jpg", ".jpeg", ".png")
video_extensions = (".mp4", ".avi", ".mov")

def annotate_image_with_detections(img, result):
    """Draws bounding boxes and labels on a PIL Image using Supervision."""
    # Convert PIL Image to numpy array for OpenCV/Supervision drawing
    img_np = np.array(img)

    detections = result.get("detections")
    if detections is not None and len(detections) > 0:
        # Create annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Build custom labels showing class ID and confidence
        labels = [
            f"animal {conf:.2f}" if cid == 0 else f"class_{cid} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]

        # Annotate the numpy image copy
        annotated_frame = box_annotator.annotate(scene=img_np.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Convert back to a PIL Image
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

def has_animal_with_conf(img):
    has_animal = False
    max_conf = 0.0

    img_array = np.array(img)
    result = detection_model.single_image_detection(img_array, det_conf_thres=DETECTION_CONF_THRESHOLD)

    annotated_img = annotate_image_with_detections(img, result)

    # PyTorch-Wildlife nests results inside the 'detections' object
    detections = result.get("detections")
    if detections is not None:
        # Access class_id and confidence arrays from the detections object
        class_ids = getattr(detections, "class_id", [])
        confidences = getattr(detections, "confidence", [])

        for cat_id, conf in zip(class_ids, confidences):
            # cat_id == 0 corresponds to animal in MegaDetector
            if cat_id == 0 and conf > 0.0:
                if conf > max_conf:
                    max_conf = conf
                if conf >= 0.15:
                    has_animal = True

    return has_animal, max_conf, annotated_img

def process_image(image_path):
    """Runs MegaDetector on a single image. Returns (has_animal, max_confidence)."""
    try:
        img = Image.open(image_path).convert("RGB")
        return has_animal_with_conf(img)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False, 0.0

def process_video(video_path):
    """Samples frames from a video file and runs MegaDetector. Returns (has_animal, max_confidence)."""
    has_animal = False
    max_conf = 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Fallback assumption

    step = int(fps)
    if step < 1:
        step = 1

    annotated_img = None

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            frame_has_animal, frame_max_conf, frame_annotated_img = has_animal_with_conf(pil_img)

            if frame_max_conf > max_conf:
                max_conf = frame_max_conf
                annotated_img = frame_annotated_img
            if frame_has_animal:
                has_animal = True

        current_frame += 1

    cap.release()
    return has_animal, max_conf, annotated_img

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

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in photo_extensions and ext not in video_extensions:
            continue

        src_path = os.path.join(cam_folder_path, filename)
        dt_obj, timestamp_str = get_file_timestamp(src_path)

        if ext in photo_extensions:
            print(f"\n\n**Processing photo:** {filename}")
            has_animal, max_conf, annotated_img = process_image(src_path)
        elif ext in video_extensions:
            print(f"\n\n**Processing video:** {filename}")
            has_animal, max_conf, annotated_img = process_video(src_path)

        # Turn max_conf into a sortable prefix (e.g., 0.923 -> conf092)
        conf_int = int(round(max_conf, 2) * 100)
        conf_prefix = f"conf{conf_int:03d}"

        # Base name with confidence first, followed by camera and timestamp
        time_slug = dt_obj.strftime("%Y-%m-%d_%H-%M-%S-%f")
        base_slug = f"{conf_prefix}_{camera_name}_{time_slug}"

        # Fallback counter check across all archive tiers
        counter = 1
        final_dest_name = base_slug
        while (os.path.exists(os.path.join(HIGH_CONF_DIR, final_dest_name)) or
               os.path.exists(os.path.join(MED_CONF_DIR, final_dest_name)) or
               os.path.exists(os.path.join(LOW_CONF_DIR, final_dest_name)) or
               os.path.exists(os.path.join(EMPTIES_DIR, final_dest_name))):
            final_dest_name = f"{base_slug}_{counter}"
            counter += 1


        new_filename = f"{final_dest_name}{ext}"
        # Route into final archive based on tiered confidence
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

        annotated_filename = f"{final_dest_name}.jpg"
        annotated_image_path = None

        if max_conf > 0 and annotated_img:
            annotated_image_path = os.path.join(ANNOTATED_DIR, annotated_filename)
            annotated_img.save(annotated_image_path)

        shutil.move(src_path, dest_path)

        current_result = {
            "timestamp": timestamp_str,
            "camera": camera_name,
            "original_filename": filename,
            "archived_filename": new_filename,
            "annotated_filename": annotated_image_path,
            "status": status_str,
            "max_confidence": round(float(max_conf), 2)
        }

        print(current_result)

        # Log everything except pure empties
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
