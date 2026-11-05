import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts import serve_preview


class QuietHandler(serve_preview.PreviewHandler):
    def log_message(self, format, *args):
        pass


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        (self.docs / "index.html").write_text("<h1>Home</h1>")
        (self.docs / "404.html").write_text("<h1>Page not found</h1>")
        (self.docs / "articles").mkdir()
        (self.docs / "articles/index.html").write_text("<h1>Articles</h1>")
        (self.root / "private.txt").write_text("outside-public-site")
        self.docs_patch = patch.object(serve_preview, "DOCS", self.docs)
        self.docs_patch.start()
        handler = partial(QuietHandler, prefix="/project/")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = "http://127.0.0.1:{}".format(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.docs_patch.stop()
        self.temporary.cleanup()

    def test_prefixed_home(self):
        with urlopen(self.origin + "/project/", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"Home", response.read())

    def test_prefix_without_slash_redirects_inside_project(self):
        with urlopen(self.origin + "/project", timeout=5) as response:
            self.assertEqual(response.geturl(), self.origin + "/project/")

    def test_nested_directory_redirect_preserves_prefix(self):
        with urlopen(self.origin + "/project/articles", timeout=5) as response:
            self.assertEqual(response.geturl(), self.origin + "/project/articles/")
            self.assertIn(b"Articles", response.read())

    def test_missing_page_has_real_404(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.origin + "/project/missing.html", timeout=5)
        self.assertEqual(error.exception.code, 404)
        self.assertIn(b"Page not found", error.exception.read())
        error.exception.close()

    def test_unprefixed_assets_are_not_accidentally_served(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.origin + "/index.html", timeout=5)
        self.assertEqual(error.exception.code, 404)
        error.exception.close()

    def test_encoded_traversal_cannot_escape_docs(self):
        with self.assertRaises(HTTPError) as error:
            urlopen(self.origin + "/project/%2e%2e/private.txt", timeout=5)
        self.assertEqual(error.exception.code, 404)
        self.assertNotIn(b"outside-public-site", error.exception.read())
        error.exception.close()

    def test_head_returns_no_body(self):
        request = Request(self.origin + "/project/", method="HEAD")
        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"")


if __name__ == "__main__":
    unittest.main()
