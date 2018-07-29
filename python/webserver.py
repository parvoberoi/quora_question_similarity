#!/usr/bin/env python
"""
Usage::./webserver.py [<port>]
"""
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from question_similarity import predict_similarity

import SocketServer
import json
import urlparse


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write("<html><body><h1>hi!</h1></body></html>")

    def do_HEAD(self):
        self._set_headers()
        
    def do_POST(self):
        # Doesn't do anything with posted data
		o = urlparse.urlparse(self.path)
		params = urlparse.parse_qs(o.query)
		self._set_headers()
		if len(params['sentence1']) != 1 or len(params['sentence2']) != 1:
			self.wfile.write("Malformed inputs provided: " + json.dumps(params))

		if predict_similarity(params['sentence1'], params['sentence2']):
			self.wfile.write("YES\n")
		else:
			self.wfile.write("NO\n")

        
def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()


