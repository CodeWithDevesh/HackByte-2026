import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import googlemaps
import re

class NavServerNode:
    def __init__(self, api_key, port=7000):
        self.app = Flask(__name__)
        CORS(self.app)
        self.gmaps = googlemaps.Client(key='AIzaSyBySlnoZoDufM1rV4yo47sCNzSj1uspbgs')
        self.port = port
        
        # State shared with the aggregator
        self.nav_state = {
            "destination": None,
            "last_instruction": "Waiting for destination..."
        }

        # Define routes inside __init__ or using decorators
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/set_destination', methods=['POST'])
        def set_dest():
            self.nav_state["destination"] = request.json.get('address')
            return jsonify({"status": "Target Locked"})

        @self.app.route('/update_and_get', methods=['POST'])
        def update_and_get():
            data = request.json
            lat, lng = data['lat'], data['lng']
            # ... (your Google Maps logic here) ...
            return jsonify({"instruction": self.nav_state["last_instruction"]})

    def run(self):
        # Running with use_reloader=False is critical when inside a thread
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)

    def start(self):
        # Start Flask in a background thread
        server_thread = threading.Thread(target=self.run, daemon=True)
        server_thread.start()

