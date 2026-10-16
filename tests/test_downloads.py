# Apache-2.0
"""Weight resolution and the cache — no network touched in these tests."""

from __future__ import annotations

import pytest

from easydetect import downloads
from easydetect.errors import DownloadError, ModelNotFoundError


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("dfine_s", "dfine-s"),
        ("D-FINE-S", "dfine-s"),
        ("DFINE_X", "dfine-x"),
        ("dfine-s.pt", "dfine-s"),
        ("dfine-l.xml", "dfine-l"),
    ],
)
def test_names_normalise_to_one_spelling(given, expected):
    assert downloads.normalize_name(given) == expected
    assert downloads.is_model_name(given)


def test_things_that_are_not_model_names_are_not_mistaken_for_one():
    assert not downloads.is_model_name("best.pt")
    assert not downloads.is_model_name("dfine-xl")
    assert not downloads.is_model_name("rtdetr-r18")


def test_the_cache_lives_under_easydetect_home(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path / "weights"))
    assert downloads.cache_dir() == tmp_path / "weights"
    assert downloads.cache_dir().is_dir()


def test_the_mirror_url_can_be_pointed_at_an_internal_copy(monkeypatch):
    assert downloads.assets_url() == downloads.DEFAULT_ASSETS_URL
    monkeypatch.setenv("EASYDETECT_ASSETS_URL", "https://mirror.internal/models/")
    assert downloads.assets_url() == "https://mirror.internal/models"


def test_an_already_cached_file_is_not_downloaded_again(tmp_path, monkeypatch):
    cached = tmp_path / "dfine-s" / "dfine-s.xml"
    cached.parent.mkdir(parents=True)
    cached.write_text("<net/>")
    monkeypatch.setattr(
        downloads.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched")
    )
    assert downloads.download("https://example.com/x.xml", cached) == cached


def test_an_unknown_variant_never_reaches_the_network():
    with pytest.raises(ModelNotFoundError, match="Known names"):
        downloads.download_ir("easydetect-r101")
    with pytest.raises(ModelNotFoundError, match="Known names"):
        downloads.download_checkpoint("easydetect-r101")


def test_a_missing_mirror_entry_suggests_training_your_own(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    monkeypatch.setattr(
        downloads, "download", lambda *a, **k: (_ for _ in ()).throw(DownloadError("HTTP 404"))
    )
    with pytest.raises(ModelNotFoundError, match="train it yourself"):
        downloads.download_ir("dfine-s")
    with pytest.raises(ModelNotFoundError, match="ImageNet backbone"):
        downloads.download_checkpoint("dfine-s")


def test_labels_txt_is_optional_when_fetching_an_ir(tmp_path, monkeypatch):
    """A mirror entry without labels.txt still yields a usable model."""
    monkeypatch.setenv("EASYDETECT_HOME", str(tmp_path))
    wanted = []

    def fake_download(url, dest, progress=True):
        wanted.append(dest.name)
        if dest.name == "labels.txt":
            raise DownloadError("HTTP 404")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("stub")
        return dest

    monkeypatch.setattr(downloads, "download", fake_download)
    xml = downloads.download_ir("dfine_s")
    assert xml.name == "dfine-s.xml"
    assert wanted == ["dfine-s.xml", "dfine-s.bin", "labels.txt"]


def _serve(routes):
    """A local HTTP server: ``routes[path] = (status, headers, body)``."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            status, headers, body = routes[self.path]
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_a_storage_etag_that_is_not_the_sha256_is_ignored(tmp_path):
    """Hugging Face redirects to storage whose ETag is another hash: a good
    file must not be thrown away for it."""
    import hashlib

    from easydetect.downloads import download

    body = b"model bytes" * 100
    sha = hashlib.sha256(body).hexdigest()
    other = hashlib.sha256(b"something else").hexdigest()
    server, base = _serve({
        "/hf": (302, {"Location": "/store", "X-Linked-Etag": f'"{sha}"'}, b""),
        "/store": (200, {"ETag": f'"{other}"'}, body),
    })
    try:
        assert download(f"{base}/hf", tmp_path / "m.bin", progress=False).read_bytes() == body
    finally:
        server.shutdown()


def test_a_file_that_does_not_match_the_linked_sha256_is_refused(tmp_path):
    import hashlib

    import pytest

    from easydetect.downloads import download
    from easydetect.errors import DownloadError

    wrong = hashlib.sha256(b"not it").hexdigest()
    server, base = _serve({
        "/hf": (302, {"Location": "/store", "X-Linked-Etag": f'"{wrong}"'}, b""),
        "/store": (200, {}, b"model bytes"),
    })
    try:
        with pytest.raises(DownloadError):
            download(f"{base}/hf", tmp_path / "m.bin", progress=False)
        assert not (tmp_path / "m.bin").exists()
    finally:
        server.shutdown()
