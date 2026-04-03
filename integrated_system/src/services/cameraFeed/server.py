import socket
import struct
import cv2
import threading
import queue
from src.core.event_bus import shared_event_bus
from src.core.events import RenderedFrameEvent

class NetworkServerNode(threading.Thread):
    def __init__(self, port: int = 9999):
        super().__init__(daemon=True)
        self.port = port
        self.frame_queue = queue.Queue(maxsize=10) 
        
        # Subscribe to the shared bus
        shared_event_bus.subscribe("rendered_frame", self.on_rendered_frame)

    def on_rendered_frame(self, event: RenderedFrameEvent):
        if not self.frame_queue.full():
            self.frame_queue.put(event.frame)

    def run(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(1)
        
        print(f"[Network] Server listening on port {self.port}...")

        while True:
            try:
                client_socket, addr = server_socket.accept()
                print(f"[Network] Client connected from {addr}")
                
                while not self.frame_queue.empty():
                    self.frame_queue.get()

                while True:
                    frame = self.frame_queue.get() 
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ret: continue
                        
                    data = buffer.tobytes()
                    message = struct.pack(">L", len(data)) + data
                    client_socket.sendall(message)
                    
            except (ConnectionResetError, BrokenPipeError):
                print("[Network] Client disconnected. Waiting for new connection...")
            except Exception as e:
                print(f"[Network] Error: {e}")
            finally:
                if 'client_socket' in locals():
                    client_socket.close()
