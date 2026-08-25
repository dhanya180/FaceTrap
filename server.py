# server.py
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

PORT = 8000
os.chdir(os.path.dirname(__file__))  # serve from current folder

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/payload.dex':
            # Optional: add a custom hash header for integrity
            with open('payload.dex', 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            # You can compute SHA-256 and send as header if needed
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'Serving on http://0.0.0.0:{PORT}/payload.dex')
    server.serve_forever()
