#!/usr/bin/env python3
"""Serve only the public static site, optionally under a project URL prefix."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, prefix="/", **kwargs):
        self.prefix = prefix
        super().__init__(*args, directory=str(DOCS), **kwargs)

    def log_request(self, code="-", size="-"):
        if not isinstance(code, int) or code >= 400:
            super().log_request(code, size)

    def map_request(self):
        parsed = urlsplit(self.path)
        if self.prefix != "/" and parsed.path == self.prefix.rstrip("/"):
            self.send_response(301)
            self.send_header(
                "Location", urlunsplit(("", "", self.prefix, parsed.query, ""))
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        if not parsed.path.startswith(self.prefix):
            self.send_error(404, "Outside the configured site prefix")
            return False
        relative = "/" + parsed.path[len(self.prefix):]
        self.path = urlunsplit(("", "", relative, parsed.query, ""))
        return True

    def do_GET(self):
        if self.map_request():
            super().do_GET()

    def do_HEAD(self):
        if self.map_request():
            super().do_HEAD()

    def send_header(self, keyword, value):
        if (
            keyword.lower() == "location"
            and self.prefix != "/"
            and value.startswith("/")
            and not value.startswith(self.prefix)
        ):
            value = self.prefix.rstrip("/") + value
        super().send_header(keyword, value)

    def send_error(self, code, message=None, explain=None):
        page = DOCS / "404.html"
        if code != 404 or not page.is_file():
            return super().send_error(code, message, explain)
        body = page.read_bytes()
        self.send_response(404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefix", default="/")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if (
        not args.prefix.startswith("/")
        or "?" in args.prefix
        or "#" in args.prefix
        or ".." in args.prefix.split("/")
        or "\\" in args.prefix
    ):
        parser.error("--prefix must be a plain absolute URL path")
    if not (DOCS / "index.html").is_file():
        parser.error("docs/index.html is missing; build the site first")
    prefix = args.prefix.rstrip("/") + "/"
    handler = partial(PreviewHandler, prefix=prefix)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(
            "Serving http://127.0.0.1:{}{}".format(server.server_port, prefix),
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nPreview server stopped.", flush=True)


if __name__ == "__main__":
    main()
