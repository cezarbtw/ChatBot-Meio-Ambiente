"""Testes offline para inspeção de protocolos de rede, telemetria e captura."""

from datetime import timedelta
import socket
import ssl
from types import SimpleNamespace

import requests

from src.collectors.g1 import G1Collector
from src.collectors.manager import build_telemetry_report, collect_from_sources
from src.network.inspector import NetworkInspector
from src.network.models import NetworkTelemetry, aggregate_protocol_telemetry
from src.network.packet_capture import (
    capture_network_traffic,
    parse_tshark_protocol_output,
)


class _DummyTlsSocket:
    def __init__(self, alpn="h2", tls_ver="TLSv1.3", cipher=("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)):
        self._alpn = alpn
        self._tls_ver = tls_ver
        self._cipher = cipher

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def selected_alpn_protocol(self):
        return self._alpn

    def version(self):
        return self._tls_ver

    def cipher(self):
        return self._cipher

    def getpeercert(self):
        return {
            "subject": ((("commonName", "g1.globo.com"),),),
            "issuer": (
                (("organizationName", "DigiCert, Inc."),),
                (("commonName", "RapidSSL G5 CA1"),),
            ),
            "notBefore": "Nov 01 00:00:00 2026 GMT",
            "notAfter": "May 31 23:59:59 2027 GMT",
        }


class _DummyRawSocket:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummySslContext:
    def __init__(self, tls_sock):
        self._tls_sock = tls_sock
        self.alpn_protocols = []

    def set_alpn_protocols(self, protos):
        self.alpn_protocols = list(protos)

    def wrap_socket(self, sock, server_hostname=None):
        if isinstance(self._tls_sock, Exception):
            raise self._tls_sock
        return self._tls_sock


def test_network_inspector_extracts_dns_tls_alpn_and_http_redirects(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type=None: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("186.192.83.7", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("186.192.83.8", port)),
        ],
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda addr, timeout=None: _DummyRawSocket(),
    )
    ctx = _DummySslContext(_DummyTlsSocket())
    monkeypatch.setattr(ssl, "create_default_context", lambda: ctx)

    hop = SimpleNamespace(
        url="http://g1.globo.com/meio-ambiente",
        status_code=301,
        raw=SimpleNamespace(version=11),
    )
    resp = SimpleNamespace(
        url="https://g1.globo.com/meio-ambiente/",
        status_code=200,
        raw=SimpleNamespace(version=11),
        history=[hop],
        elapsed=timedelta(milliseconds=145),
        headers={"Content-Type": "text/html; charset=UTF-8"},
    )

    inspector = NetworkInspector(timeout=5.0)
    tel = inspector.inspect_response(resp, total_ms=190.5)

    assert tel.host == "g1.globo.com"
    assert tel.resolved_ips == ["186.192.83.7", "186.192.83.8"]
    assert tel.http_version == "HTTP/1.1"
    assert tel.alpn_protocol == "h2"
    assert tel.tls_version == "TLSv1.3"
    assert tel.cipher_suite == "TLS_AES_256_GCM_SHA384"
    assert tel.certificate is not None
    assert tel.certificate.verified is True
    assert tel.certificate.issuer_org == "DigiCert, Inc."
    assert tel.status_code == 200
    assert len(tel.redirect_chain) == 1
    assert tel.redirect_chain[0].status_code == 301
    assert tel.ttfb_ms == 145.0
    assert tel.total_ms == 190.5
    assert tel.error is None
    assert ctx.alpn_protocols == ["h2", "http/1.1"]


def test_network_inspector_handles_dns_and_tls_failures_gracefully(monkeypatch):
    def _fail_dns(host, port, type=None):
        raise socket.gaierror("DNS resolution failed")

    monkeypatch.setattr(socket, "getaddrinfo", _fail_dns)
    inspector = NetworkInspector(cache_transport_per_host=False)
    dns_fail = inspector.inspect_transport("invalido.local", 443, use_tls=True)
    assert dns_fail["resolved_ips"] == []
    assert "Falha DNS" in dns_fail["error"]

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type=None: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        ],
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda addr, timeout=None: _DummyRawSocket(),
    )
    monkeypatch.setattr(
        ssl,
        "create_default_context",
        lambda: _DummySslContext(ssl.SSLError("certificate verify failed")),
    )

    tls_fail = inspector.inspect_transport("expired.local", 443, use_tls=True)
    assert tls_fail["resolved_ips"] == ["127.0.0.1"]
    assert tls_fail["certificate"].verified is False
    assert "Falha TLS/certificado" in tls_fail["error"]


def test_network_inspector_inspect_url_handles_request_exception(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type=None: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))
        ],
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda addr, timeout=None: _DummyRawSocket(),
    )
    monkeypatch.setattr(
        ssl,
        "create_default_context",
        lambda: _DummySslContext(_DummyTlsSocket(alpn="http/1.1", tls_ver="TLSv1.2")),
    )

    class _FailingSession:
        def request(self, **kwargs):
            raise requests.Timeout("timed out")

    tel = NetworkInspector().inspect_url("https://exemplo.org/feed", session=_FailingSession())
    assert tel.host == "exemplo.org"
    assert tel.alpn_protocol == "http/1.1"
    assert tel.tls_version == "TLSv1.2"
    assert "timed out" in tel.error


def test_aggregate_protocol_telemetry_and_collector_integration(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type=None: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port))
        ],
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda addr, timeout=None: _DummyRawSocket(),
    )
    monkeypatch.setattr(
        ssl,
        "create_default_context",
        lambda: _DummySslContext(_DummyTlsSocket()),
    )

    html = """
    <html><body>
      <div class="feed-post">
        <a class="feed-post-body-title" href="https://g1.globo.com/meio-ambiente/noticia/2026/11/25/mata.ghtml">Mata recuperada</a>
        <div class="feed-post-body-resumo">Área verde ampliada.</div>
      </div>
    </body></html>
    """

    class _FakeResp:
        url = G1Collector.DEFAULT_URL
        status_code = 200
        raw = SimpleNamespace(version=11)
        history = []
        elapsed = timedelta(milliseconds=80)
        headers = {"Content-Type": "text/html"}
        content = html.encode("utf-8")
        text = html

        def raise_for_status(self):
            return None

    class _MockSession:
        def get(self, url, params=None, timeout=None):
            return _FakeResp()

    collector = G1Collector(session=_MockSession(), inspect_network=True)
    items, reports = collect_from_sources(collectors=[collector])

    assert len(items) == 1
    assert "ALPN h2" in items[0].protocol
    assert "TLSv1.3" in items[0].protocol
    assert items[0].network_telemetry["resolved_ips"] == ["10.0.0.1"]
    assert reports[0].source_telemetry is not None

    report = build_telemetry_report(reports, items=items)
    assert report["summary"]["total_inspected"] == 1
    assert report["summary"]["alpn_protocols"] == {"h2": 1}
    assert report["summary"]["tls_versions"] == {"TLSv1.3": 1}

    agg = aggregate_protocol_telemetry(
        [
            NetworkTelemetry(url="https://a", final_url="https://a", scheme="https", host="a", port=443, http_version="HTTP/1.1", tls_version="TLSv1.3", dns_ms=10.0, ttfb_ms=100.0),
            NetworkTelemetry(url="http://b", final_url="http://b", scheme="http", host="b", port=80, http_version="HTTP/1.1", dns_ms=20.0, ttfb_ms=200.0, error="timeout"),
        ]
    )
    assert agg["schemes"] == {"https": 1, "http": 1}
    assert agg["errors_count"] == 1
    assert agg["avg_dns_ms"] == 15.0
    assert agg["avg_ttfb_ms"] == 150.0


def test_packet_capture_parser_and_graceful_unavailable(monkeypatch):
    counts = parse_tshark_protocol_output(
        [
            "eth:ethertype:ip:tcp:tls",
            "eth:ethertype:ip:udp:dns",
            "eth:ethertype:ip:tcp:http",
        ]
    )
    assert counts == {"TCP": 2, "TLS": 1, "UDP": 1, "DNS": 1, "HTTP": 1}

    monkeypatch.setattr("src.network.packet_capture.detect_capture_backend", lambda: None)
    summary = capture_network_traffic(duration_seconds=2)
    assert summary.available is False
    assert summary.error is not None
