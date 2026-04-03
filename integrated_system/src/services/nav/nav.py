import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import googlemaps

class NavServerNode:
    def __init__(self, api_key: str, port: int = 7000):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # FIX 1: Use the passed api_key variable, NOT a hardcoded string!
        if not api_key:
            raise ValueError("Google Maps API key is required to start the NavServerNode.")
        self.gmaps = googlemaps.Client(key=api_key)
        
        self.port = port
        
        # State shared with the aggregator
        self.nav_state = {
            "destination": None,
            "last_instruction": "Waiting for destination..."
        }

        # Define routes
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/set_destination', methods=['POST'])
        def set_dest():
            # FIX 2: Safely parse JSON
            data = request.get_json()
            if not data or 'address' not in data:
                return jsonify({"error": "Missing 'address' in request body"}), 400
                
            self.nav_state["destination"] = data.get('address')
            print(f"[NavServer] Destination locked to: {self.nav_state['destination']}")
            return jsonify({"status": "Target Locked", "destination": self.nav_state["destination"]}), 200

        @self.app.route('/update_and_get', methods=['POST'])
        def update_and_get():
            data = request.get_json()
            
            # Safely check if lat/lng exist before trying to use them
            if not data or 'lat' not in data or 'lng' not in data:
                return jsonify({"error": "Missing 'lat' or 'lng' in request body"}), 400
                
            lat, lng = data['lat'], data['lng']
            
            # ... (your Google Maps logic here) ...
            
            return jsonify({"instruction": self.nav_state["last_instruction"]}), 200

    def run(self):
        print(f"[NavServer] Starting Flask API on port {self.port}...")
        # Running with use_reloader=False is critical when inside a thread
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)

    def start(self):
        # Start Flask in a background thread
        server_thread = threading.Thread(target=self.run, daemon=True)
        server_thread.start()
