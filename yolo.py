import cv2
import requests
import numpy as np
from ultralytics import YOLO
import time
import threading

# --- CONFIGURATION ---
NODE_RED_URL = "http://127.0.0.1:1880/motion" # Node-Red URL, some devices may need to change this to their local IP
VIDEO_SOURCE = "Footage/delivery.mp4" 
CONFIDENCE_THRESHOLD = 0.5
LOITER_LIMIT = 2.0 

# --- MEMORY ---
zone_timers = {}    # Stopwatch for current loitering
track_states = {}   # Remembers "CRITICAL" vs "ALERT" to stop spam
last_durations = {} # <--- NEW: Remembers the final time when they leave Danger zone

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)

alert_zone_polygon = None
danger_zone_polygon = None

def send_request(payload):
    def _send():
        try:
            requests.post(NODE_RED_URL, json=payload)
            print(f">> SENT: {payload['status']}")
        except Exception as e:
            print(f"Error: {e}")
    threading.Thread(target=_send, daemon=True).start()

def is_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

while True:
    ret, frame = cap.read()
    if not ret: break

    height, width = frame.shape[:2]

    # --- 1. DEFINE ZONES ---
    if danger_zone_polygon is None:
        split_y = int(height * 0.70) 
        split_width_x = int(width * 0.68)

        # Danger Zone (Top)
        alert_zone_polygon = np.array([
            [0, split_y*0.6],                   
            [split_width_x, split_y],           
            [int(width * 0.70), int(height * 0.38)], 
            [0, int(height * 0.3)]              
        ], np.int32).reshape((-1, 1, 2))

        # Alert Zone (Bottom)
        danger_zone_polygon = np.array([
            [0, height],                      
            [int(width * 0.65), height],      
            [split_width_x, split_y],         
            [0, split_y*0.6]                  
        ], np.int32).reshape((-1, 1, 2))


    # --- 2. TRACKING ---
    results = model.track(frame, persist=True, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=[0])

    status = "CLEAR"
    status_color = (0, 255, 0) # Green

    cv2.polylines(frame, [alert_zone_polygon], isClosed=True, color=(0, 255, 255), thickness=2) 
    cv2.polylines(frame, [danger_zone_polygon], isClosed=True, color=(0, 0, 255), thickness=2)   
    cv2.putText(frame, "ALERT", (10, int(height * 0.35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(frame, "DANGER", (10, int(height * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    current_ids = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            feet_x, feet_y = int((x1 + x2) / 2), int(y2)
            current_ids.append(track_id)
            
            in_danger = is_inside_polygon((feet_x, feet_y), danger_zone_polygon)
            in_alert = is_inside_polygon((feet_x, feet_y), alert_zone_polygon)
            
            box_color = (0, 255, 0)
            last_sent_mode = track_states.get(track_id, "SAFE")

            # --- DANGER ZONE ---
            if in_danger:
                status = "People in danger zone"
                status_color = (0, 0, 255) # Red
                box_color = (0, 0, 255)

                if track_id not in zone_timers:
                    zone_timers[track_id] = time.time()
                
                # Calculate Duration
                elapsed_time = time.time() - zone_timers[track_id]
                
                # Save this duration instantly (So we don't lose it if they step out)
                last_durations[track_id] = elapsed_time

                if elapsed_time > LOITER_LIMIT:
                    duration_str = f"{elapsed_time:.1f} second(s)"
                    status = f"Loitering in front of main door for {duration_str}"

                    if last_sent_mode != "CRITICAL":
                        payload = {"status": "CRITICAL Mode", "msg": status}
                        send_request(payload)
                        track_states[track_id] = "CRITICAL"

            # --- ALERT ZONE ---
            elif in_alert:
                status = "People around the house"
                status_color = (0, 255, 255) # Yellow
                box_color = (0, 255, 255)

                if last_sent_mode != "ALERT" and last_sent_mode != "CRITICAL":
                    payload = {"status": "ALERT Mode", "msg": status}
                    send_request(payload)
                    track_states[track_id] = "ALERT"

                # If they just left Danger Zone, the timer is deleted here.
                # But we already saved the time in 'last_durations' above!
                if track_id in zone_timers: del zone_timers[track_id]

            # --- SAFE ZONE (Exit) ---
            else:
                # If they were previously dangerous/alert, send final report
                if last_sent_mode in ["ALERT", "CRITICAL"]:
                    
                    # RETRIEVE SAVED DURATION (Fix for 0.0s)
                    final_time = last_durations.get(track_id, 0.0)
                    
                    msg = f"Target left zone. Total duration loitering in front of house: {final_time:.1f} second(s)"
                    payload = {"status": "SAFE Mode", "msg": msg}
                    send_request(payload)
                    
                    track_states[track_id] = "SAFE"

                if track_id in zone_timers: del zone_timers[track_id]


            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (feet_x, feet_y), 5, (255, 255, 255), -1)

    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    # Cleanup Memory
    for track_id in list(track_states.keys()):
        if track_id not in current_ids:
            del track_states[track_id]
            if track_id in zone_timers: del zone_timers[track_id]
            if track_id in last_durations: del last_durations[track_id]

    cv2.imshow("Top-Bottom Zone Split", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()