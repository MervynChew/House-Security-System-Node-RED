# Note
1. The main program is just yolo.py
2. detect.py is the old version
3. yolo.py will use content in the Footage for audios and videos

# Setup before Running Python Script
1. Change Node-RED URL in yolo.py (Line  9), some devices use http://127.0.0.1:1880/motion, some use http://localhost:1880/motion.
2. Get API Key from https://m.me/api.callmebot, type "create apikey" with the chatbot, then copy and change the API key in JSON file (Line 442 and Line 527) in "Flow" folder.
3. Now you are able to receive notifications on your messenger.
4. Change directory for Alarm.wav in JSON file (Line 282).
5. Choose Clipboard on Node-RED and import the JSON file in "Flow" folder.

# Run Python Script
1. Ensure Node-RED is open on the computer with the JSON file.
2. Type inside the command prompt or terminal: python yolo.py
3. A video is activated and can refer to the Node-RED.
4. Console will show the detection result.
5. Audio, house light changes, messenger notifications and history logs recording will be triggered.