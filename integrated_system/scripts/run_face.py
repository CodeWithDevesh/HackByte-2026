import time
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so `import src...` works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import your decoupled nodes
from src.hardware.camera import CameraNode
from src.models.face_recognition.model import build_default_face_pipeline
from src.services.cameraFeed.server import NetworkServerNode

def main():
    print("=============================================")
    print("  Starting Decoupled AI Vision Pipeline...   ")
    print("=============================================")

    try:
        # 1. Initialize the AI & Drawing Nodes 
        # (They automatically subscribe to the shared_event_bus internally)
        print("\n[*] Initializing AI Models and TTS...")
        face_node, aggregator_node = build_default_face_pipeline()

        # 2. Start the Network Server Thread
        # (It automatically subscribes to "rendered_frame" events)
        print("[*] Starting TCP Video Server...")
        server = NetworkServerNode(port=9999)
        server.start()

        # 3. Start the Camera Thread
        # (We start this LAST. If we start it first, it will blast frames 
        # into the void before YOLO has finished loading into memory!)
        print("[*] Warming up Camera Hardware...")
        camera = CameraNode(camera_index=0)
        camera.start()

        print("\n[+] System is fully operational!")
        print("[+] Waiting for client to connect to view feed...")
        print("    (Press Ctrl+C to shut down gracefully)")
        print("=============================================\n")

        # 4. Keep the main thread alive
        # The while loop is required because camera and server are running 
        # on background daemon threads. If main() ends, the script dies.
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n[-] Ctrl+C detected. Initiating graceful shutdown...")
    except Exception as e:
        print(f"\n[!] Fatal Error in main loop: {e}")
    finally:
        print("[-] Pipeline terminated.")
        os._exit(0)

if __name__ == "__main__":
    main()
