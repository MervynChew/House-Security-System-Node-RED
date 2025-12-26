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
LOITER_LIMIT_DANGER = 2.0 
LOITER_LIMIT_ALERT = 2.0  

# --- MEMORY ---
danger_zone_timers = {}    # Stopwatch for current loitering in Danger Zone
alert_zone_timers = {}     # Stopwatch for current loitering in Alert Zone
track_states = {}          # Remembers "CRITICAL" vs "ALERT" to stop spam
danger_total_time = {}     # Total accumulated time in Danger Zone
alert_total_time = {}      # Total accumulated time in Alert Zone

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

                if track_id not in danger_zone_timers:
                    danger_zone_timers[track_id] = time.time()
                
                # Calculate Duration
                danger_elapsed_time = time.time() - danger_zone_timers[track_id]
                
                if danger_elapsed_time > LOITER_LIMIT_DANGER:
                    duration_str = f"{danger_elapsed_time:.1f} second(s)"
                    status = f"Loitering in front of house for {duration_str}"

                    if last_sent_mode != "CRITICAL":
                        payload = {"status": "CRITICAL Mode", "msg": status}
                        send_request(payload)
                        track_states[track_id] = "CRITICAL"
                
                # Save in alert zone duration instantly (So we don't lose it if they step out)
                if track_id in alert_zone_timers:
                    alert_total_time[track_id] = alert_total_time.get(track_id, 0) + (time.time() - alert_zone_timers[track_id])
                    del alert_zone_timers[track_id]

            # --- ALERT ZONE ---
            elif in_alert:
                status = "People around the house"
                status_color = (0, 255, 255) # Yellow
                box_color = (0, 255, 255)

                # Start Alert Timer
                if track_id not in alert_zone_timers:
                    alert_zone_timers[track_id] = time.time()
                
                # Calculate Duration
                alert_elapsed_time = time.time() - alert_zone_timers[track_id]
                
                if alert_elapsed_time > LOITER_LIMIT_ALERT:
                    duration_str = f"{alert_elapsed_time:.1f} second(s)"
                    status = f"Loitering in alert zone for {duration_str}"

                    # Only send "ALERT" event when someone moves from SAFE to ALERT, and from CRITICAL to ALERT
                    if last_sent_mode != "ALERT":
                        # Only play audio if moving from SAFE to ALERT
                        should_play = 1 if last_sent_mode != "CRITICAL" else 0
        
                        payload = {
                            "status": "ALERT Mode", 
                            "play_audio": should_play,
                            "msg": status,
                        }
                        send_request(payload)
                        # Update state
                        track_states[track_id] = "ALERT"

                # Save in danger zone duration instantly (So we don't lose it if they step out)
                if track_id in danger_zone_timers:
                    danger_total_time[track_id] = danger_total_time.get(track_id, 0) + (time.time() - danger_zone_timers[track_id])
                    del danger_zone_timers[track_id]

            # --- SAFE ZONE (Exit) ---
            else:
                # If they were previously dangerous/alert, send final report
                if last_sent_mode in ["ALERT", "CRITICAL"]:
                    
                    final_danger_time = danger_total_time.get(track_id, 0) + (time.time() - danger_zone_timers.get(track_id, time.time()))
                    final_alert_time = alert_total_time.get(track_id, 0) + (time.time() - alert_zone_timers.get(track_id, time.time()))
                    
                    # Get type of zone someone is in based on last sent mode
                    zone_name = "alert zone" if last_sent_mode == "ALERT" else "danger zone"

                    # Newline for each duration
                    msg1 = f"Target left.<br>"
                    msg2 = f"Total duration loitering in front of house: {max(0, final_danger_time):.1f} second(s)<br>"
                    msg3 = f"Total duration loitering in alert zone: {max(0, final_alert_time):.1f} second(s)"
                    payload = {"status": "SAFE Mode", "msg": msg1 + msg2 + msg3}
                    send_request(payload)
                    
                    track_states[track_id] = "SAFE"
                    # Reset total times for alert and danger zones
                    danger_total_time[track_id] = 0
                    alert_total_time[track_id] = 0

                if track_id in danger_zone_timers: del danger_zone_timers[track_id]
                if track_id in alert_zone_timers: del alert_zone_timers[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (feet_x, feet_y), 5, (255, 255, 255), -1)

    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    # Cleanup Memory
    for track_id in list(track_states.keys()):
        if track_id not in current_ids:
            del track_states[track_id]
            if track_id in danger_zone_timers: del danger_zone_timers[track_id]

    cv2.imshow("Top-Bottom Zone Split", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()