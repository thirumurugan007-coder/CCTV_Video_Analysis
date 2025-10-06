import argparse
import json
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Lazy import Ultralytics YOLO
YOLO_OK = True
try:
    from ultralytics import YOLO
except Exception:
    YOLO_OK = False

# COCO subset map (common classes)
COCO_NAME_TO_ID: Dict[str, int] = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5, "train": 6, "truck": 7, "boat": 8,
    "traffic light": 9, "fire hydrant": 10, "stop sign": 12, "bench": 14, "bird": 15, "cat": 16, "dog": 17,
    "backpack": 24, "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28,
    "bottle": 39, "wine glass": 40, "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
    "chair": 56, "couch": 57, "potted plant": 58, "bed": 59, "dining table": 60, "toilet": 61,
    "tv": 62, "laptop": 63, "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67,
    "microwave": 69, "oven": 70, "toaster": 71, "sink": 72, "refrigerator": 73,
    "book": 84, "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88, "hair drier": 89, "toothbrush": 90
}
ALIASES: Dict[str, str] = {
    "bike": "bicycle",
    "bikes": "bicycle",
    "motorbike": "motorcycle",
    "tvmonitor": "tv",
    "mobile": "cell phone",
    "phone": "cell phone",
    "hand bag": "handbag",
    "plant": "potted plant",
    "table": "dining table",
    # approximate mapping
    "box": "suitcase",
    "package": "suitcase",
}
COCO_ID_TO_NAME: Dict[int, str] = {v: k for k, v in COCO_NAME_TO_ID.items()}

VEHICLE_IDS = [COCO_NAME_TO_ID[k] for k in ["bicycle", "car", "motorcycle", "bus", "truck"]]
HUMAN_IDS = [COCO_NAME_TO_ID["person"]]

@dataclass
class Event:
    event_type: str
    labels: List[str]
    start_s: float
    end_s: float
    confidence: float

def name_to_ids(names: List[str]) -> List[int]:
    ids: List[int] = []
    for n in names:
        key = n.strip().lower()
        if not key:
            continue
        if key in ALIASES:
            key = ALIASES[key]
        if key in COCO_NAME_TO_ID:
            ids.append(COCO_NAME_TO_ID[key])
        else:
            print(f"[WARN] Unknown object name '{n}'. Skipping. (Tip: try suitcase/backpack/handbag for 'box').")
    return sorted(set(ids))

def within_rect(pt: Tuple[int, int], rect: Optional[Tuple[int, int, int, int]]) -> bool:
    if rect is None:
        return True
    x, y = pt
    rx, ry, rw, rh = rect
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

def prompt_if_missing(args: argparse.Namespace) -> argparse.Namespace:
    # Prompt for input video/RTSP if missing
    while not args.input:
        candidate = input("Enter video path or RTSP URL: ").strip().strip('"')
        if not candidate:
            continue
        if candidate.lower().startswith(("rtsp://", "rtsps://")) or os.path.exists(candidate):
            args.input = candidate
        else:
            print("[INFO] Path does not exist. If you intended an RTSP URL, ensure it starts with rtsp://")
    # Prompt for task if missing
    valid_tasks = ["movement", "humans", "vehicles", "objects"]
    while not args.task:
        t = input("Choose task [movement|humans|vehicles|objects]: ").strip().lower()
        if t in valid_tasks:
            args.task = t
        else:
            print("[INFO] Invalid task. Please choose one of:", ", ".join(valid_tasks))
    if args.task == "objects" and (not args.objects or not args.objects.strip()):
        obj = ""
        while not obj:
            obj = input("Enter comma-separated object names (e.g., car,bike,box): ").strip()
        args.objects = obj
    return args

def main():
    ap = argparse.ArgumentParser(description="CCTV Video Analyzer (YOLO11) with real-time playback control.")
    ap.add_argument("--input", default=None, help="Video file path or RTSP URL")
    ap.add_argument("--task", default=None, choices=["movement", "humans", "vehicles", "objects"], help="Analysis task")
    ap.add_argument("--objects", default="", help="Comma-separated object names (when task=objects)")
    ap.add_argument("--imgsz", type=int, default=512, help="Detector image size (512 CPU, 640 GPU)")
    ap.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    ap.add_argument("--process-fps", type=float, default=8.0, help="Max processing FPS (saves CPU)")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame (offline speedup)")
    ap.add_argument("--idle-gap", type=float, default=3.0, help="Seconds of inactivity to close an event")
    ap.add_argument("--min-motion-area", type=int, default=1200, help="Min contour area (pixels) to consider motion")
    ap.add_argument("--gate-by-motion", type=int, default=0, help="0 = YOLO always runs; 1 = only when motion detected")
    ap.add_argument("--roi-rect", type=int, nargs=4, metavar=("x", "y", "w", "h"), help="ROI rectangle (x y w h)")
    ap.add_argument("--events", default="", help="Optional events.jsonl output")
    ap.add_argument("--summary", default="", help="Optional summary.json output")
    ap.add_argument("--annot", default="", help="Optional annotated MP4 output")
    ap.add_argument("--yolo-model", default="yolo11n.pt", help="YOLO model (fallback yolov8n.pt if unavailable)")
    ap.add_argument("--show", type=int, default=1, help="1 = display live annotated frames; 0 = no window")
    ap.add_argument("--nms-threshold", type=float, default=0.45, help="NMS threshold for post-processing")
    ap.add_argument("--display-fps", type=float, default=0.0, help="Playback FPS (0=auto: source fps adjusted for stride/process)")
    ap.add_argument("--realtime", type=int, default=1, help="1 = sleep to match display FPS; 0 = as fast as possible")
    args = ap.parse_args()

    # Prompt interactively if needed
    args = prompt_if_missing(args)

    # Load model if needed
    target_ids: List[int] = []
    if args.task == "humans":
        target_ids = HUMAN_IDS
    elif args.task == "vehicles":
        target_ids = VEHICLE_IDS
    elif args.task == "objects":
        names = [s for s in args.objects.split(",") if s.strip()]
        if not names:
            raise SystemExit("For --task objects, provide --objects name1,name2,... (e.g., car,bike,backpack)")
        target_ids = name_to_ids(names)

    model = None
    if args.task in ("humans", "vehicles", "objects"):
        if not YOLO_OK:
            raise SystemExit("Ultralytics not installed. Run: pip install ultralytics")
        try:
            print(f"[INFO] Loading YOLO model: {args.yolo_model}")
            model = YOLO(args.yolo_model)
        except Exception as e:
            print(f"[WARN] Failed to load {args.yolo_model} ({e}); trying yolov8n.pt")

            try:
                model = YOLO("yolov8n.pt")
            except Exception as e:
                raise SystemExit(f"Failed to load fallback yolov8n.pt model: {e}")
            model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit("Failed to open input. Check path/RTSP and FFmpeg installation.")
    print("[INFO] Video opened.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    frame_count_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(frame_count_prop) if frame_count_prop and frame_count_prop > 0 else 0

    # Compute display FPS and delay
    if args.display_fps > 0:
        disp_fps = float(args.display_fps)
    else:
        # Auto: if process-fps caps processing, use min(source, process-fps); adjust for stride
        disp_fps = fps
        if args.process_fps > 0:
            disp_fps = min(disp_fps, args.process_fps)
    if args.stride > 1:
        # Showing fewer frames => reduce display fps to maintain real-time wall-clock speed
        disp_fps = max(1.0, disp_fps / args.stride)
    delay_ms = max(1, int(1000.0 / disp_fps)) if args.realtime else 1
    print(f"[INFO] Playback target: ~{disp_fps:.1f} FPS (delay {delay_ms} ms)")

    # Optional writer
    writer = None
    if args.annot and W > 0 and H > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.annot, fourcc, fps, (W, H))
        if not writer.isOpened():
            print("[WARN] Failed to open annot writer; continuing without video output.")
            writer = None

    # Display window
    if args.show:
        cv2.namedWindow("CCTV Analyzer", cv2.WINDOW_NORMAL)

    # Motion background subtractor
    bg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=16, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Timers
    last_proc = 0.0
    interval = 1.0 / max(1e-3, args.process_fps)
    frame_idx = 0

    # Events
    events: List[Event] = []
    ev_open: Optional[Event] = None
    last_active_s = 0.0
    label_counts: Dict[str, int] = {}
    roi_rect = tuple(args.roi_rect) if args.roi_rect else None

    def append_event(e: Event):
        events.append(e)

    def render_output(disp: np.ndarray) -> bool:
        """Render to window and/or writer. Returns False if user requested quit."""
        if writer:
            writer.write(disp)
        if args.show:
            cv2.imshow("CCTV Analyzer", disp)
            key = (cv2.waitKey(delay_ms) & 0xFF)
            if key in (27, ord('q')):
                return False
        elif args.realtime and delay_ms > 0:
            # If not showing a window, still sleep to keep normal speed
            time.sleep(delay_ms / 1000.0)
        return True

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1

            # Optionally skip frames for speed
            if args.stride > 1 and (frame_idx % args.stride != 0):
                if not render_output(frame):
                    break
                continue

            now = time.time()
            if now - last_proc < interval:
                if not render_output(frame):
                    break
                continue
            last_proc = now

            t_s = (frame_idx / fps) if fps else 0.0

            # Motion pipeline (used for gating and visualization)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi_rect:
                rx, ry, rw, rh = roi_rect
                mask = np.zeros_like(gray)
                cv2.rectangle(mask, (rx, ry), (rx + rw, ry + rh), 255, -1)
                masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            else:
                masked_gray = gray

            fg = bg.apply(masked_gray)
            fg = cv2.medianBlur(fg, 5)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
            fg = cv2.dilate(fg, kernel, iterations=2)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= args.min_motion_area)
            motion = motion_area > 0

            # Detection
            activity = False
            labels_this_frame: List[str] = []
            conf_this_frame = 0.0
            boxes_to_draw: List[Tuple[int, int, int, int, str, float]] = []

            if args.task == "movement":
                activity = motion
            else:
                if args.gate_by_motion and not motion:
                    activity = False
                else:
                    res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False, classes=target_ids,iou=args.nms_threshold)
                    pred = res[0]
                    if pred.boxes is not None and pred.boxes.xyxy is not None:
                        xyxy = pred.boxes.xyxy.cpu().numpy().astype(int)
                        cls_arr = pred.boxes.cls.cpu().numpy().astype(int)
                        conf_arr = pred.boxes.conf.cpu().numpy()
                        kept = 0
                        confs_accum: List[float] = []
                        for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls_arr, conf_arr):
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            if not within_rect((cx, cy), roi_rect):
                                continue
                            name = COCO_ID_TO_NAME.get(int(cid), str(int(cid)))
                            labels_this_frame.append(name)
                            label_counts[name] = label_counts.get(name, 0) + 1
                            confs_accum.append(float(cf))
                            kept += 1
                            boxes_to_draw.append((int(x1), int(y1), int(x2), int(y2), name, float(cf)))
                        if kept > 0:
                            activity = True
                            conf_this_frame = float(np.mean(confs_accum)) if confs_accum else 0.0

            # Event aggregation
            if activity:
                if ev_open is None:
                    ev_open = Event(
                        event_type=args.task if args.task != "objects" else "objects",
                        labels=[],
                        start_s=t_s,
                        end_s=t_s,
                        confidence=0.0,
                    )
                ev_open.end_s = t_s
                last_active_s = t_s
                if args.task != "movement" and labels_this_frame:
                    ev_open.labels.extend(labels_this_frame)
                    if conf_this_frame > 0:
                        ev_open.confidence = conf_this_frame if ev_open.confidence == 0.0 else (0.5 * ev_open.confidence + 0.5 * conf_this_frame)
            else:
                if ev_open is not None and (t_s - last_active_s) >= args.idle_gap:
                    ev_open.labels = sorted(set(ev_open.labels))
                    append_event(ev_open)
                    ev_open = None

            # Draw
            disp = frame.copy()
            if roi_rect:
                rx, ry, rw, rh = roi_rect
                cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                cv2.putText(disp, "ROI", (rx, max(0, ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            if args.task == "movement":
                if motion:
                    cv2.putText(disp, "MOTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
            else:
                for (x1, y1, x2, y2, name, cf) in boxes_to_draw:
                    color = (0, 255, 0) if name == "person" else (255, 0, 0)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(disp, f"{name} {cf:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if ev_open is not None:
                cv2.putText(disp, "EVENT OPEN", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

            # Output
            if not render_output(disp):
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        if ev_open is not None:
            ev_open.labels = sorted(set(ev_open.labels))
            append_event(ev_open)

    # Write events.jsonl
    if args.events:
        with open(args.events, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps({
                    "event_type": ev.event_type,
                    "labels": ev.labels,
                    "start_s": round(ev.start_s, 3),
                    "end_s": round(ev.end_s, 3),
                    "duration_s": round(ev.end_s - ev.start_s, 3),
                    "confidence": round(ev.confidence, 3),
                }) + "\n")

    # Summary
    total_duration = (frame_count / fps) if frame_count and fps else (frame_idx / fps if fps else 0.0)
    avg_conf = float(np.mean([e.confidence for e in events if e.confidence > 0])) if events else 0.0
    avg_ev_dur = float(np.mean([e.end_s - e.start_s for e in events])) if events else 0.0
    summary = {
        "task": args.task,
        "events_total": len(events),
        "label_counts": {k: int(v) for k, v in label_counts.items()},
        "avg_confidence": round(avg_conf, 3),
        "avg_event_duration_s": round(avg_ev_dur, 3),
        "estimated_video_duration_s": round(total_duration, 3),
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()