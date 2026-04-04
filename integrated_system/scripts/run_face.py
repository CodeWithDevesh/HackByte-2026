import time
import os
import sys
import threading
import cv2
from pathlib import Path
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
DIRECTION_API = os.getenv("DIRECTION_API")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core & Infrastructure Imports
from src.core.event_bus import shared_event_bus
from src.core.aggregator import AggregatorNode 
from src.core.voice_assistant import VoiceAssistant
from src.hardware.camera import CameraNode
from src.services.cameraFeed.server import NetworkServerNode
from src.services.nav.nav import NavServerNode

# Model Builders
from src.models.weapon_detection.model import build_default_weapon_node
from src.models.ocr.model import build_default_ocr_node 
from src.models.face_recognition.model import build_default_face_node

def main():
    print("=============================================")
    print("   AI VISION SYSTEM: INTEGRATED PIPELINE     ")
    print("       (Voice + Vision + Navigation)         ")
    print("=============================================")

    try:
        # 1. Initialize Vision Models
        print("[*] Initializing Face Recognition...")
        face_node = build_default_face_node()

        print("[*] Initializing Weapon Detection...")
        weapon_node = build_default_weapon_node()

        print("[*] Initializing OCR System...")
        ocr_node = build_default_ocr_node()

        # 2. Initialize Central Logic & Voice
        print("[*] Initializing Central Aggregator...")
        # Tracks results from the various visual processing nodes
        aggregator = AggregatorNode(expected_models=["FaceModel", "WeaponModel", "OCRModel"])

        print("[*] Initializing Voice Assistant (Headset Mode)...")
        assistant = VoiceAssistant()
        assistant.start()

        # 3. Start Services (Networking & Nav)
        print(f"[*] Starting Navigation API Server...")
        nav_node = NavServerNode(api_key=DIRECTION_API)
        nav_node.start()

        print("[*] Starting TCP Video Server (Port 9999)...")
        server = NetworkServerNode(port=9999)
        server.start()

        # 4. Start Hardware
        print("[*] Warming up Camera Hardware...")
        camera = CameraNode(camera_index=0)
        camera.start()

        print("\n[+] SYSTEM FULLY OPERATIONAL")
        print("[+] Press your Headset Button to give a command.")
        print("[+] Press 'q' on the monitor window to exit.")
        print("=============================================\n")

        # 5. Main Loop for local display and persistence
        while True:
            # Check for local frame display
            frame = getattr(camera, 'frame', None) or getattr(camera, '_frame', None)
            
            if frame is not None:
                cv2.imshow("Security Monitor Feed", frame)
            
            # Use 'q' to break the loop or let it run indefinitely
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Small sleep to prevent high CPU usage in the management loop
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\n[-] Shutdown initiated by user...")
    except Exception as e:
        print(f"\n[!] Fatal Error: {e}")
    finally:
        # Graceful Cleanup
        print("[-] Cleaning up resources...")
        if 'assistant' in locals():
            assistant.stop()
        
        cv2.destroyAllWindows()
        print("[-] Pipeline terminated.")
        os._exit(0)

if __name__ == "__main__":
    main()