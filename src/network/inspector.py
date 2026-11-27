"""Inspetor real de DNS, TCP, TLS/certificado, ALPN e HTTP."""

from datetime import datetime, timezone
import socket
import ssl
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from src.network.models import (
    CertificateInfo,
    NetworkTelemetry,
    RedirectHop,
)


def _format_http_version(raw_version: Any) -> Optional[str]:
    if raw_version is None:
        return None
    if isinstance(raw_version, str):
        return raw_version
    mapping = {10: "HTTP/1.0", 11: "HTTP/1.1", 20: "HTTP/2"}
    return mapping.get(int(raw_version), f"HTTP/{raw_version}")


def _extract_rdn_field(rdn_seq: Any, field_name: str) -> Optional[str]:
    if not rdn_seq:
        return None
    for rdn in rdn_seq:
        for key, value in rdn:
            if key == field_name:
                return str(value)
    return None


def _parse_cert_timestamp(raw_date: Optional[str]) -> Optional[datetime]:
    if not raw_date:
        return None
    try:
        seconds = ssl.cert_time_to_seconds(raw_date)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def _build_certificate_info(
    peercert: Optional[Dict[str, Any]], verified: bool = True
) -> Optional[CertificateInfo]:
    if not peercert:
        return CertificateInfo(verified=verified) if not verified else None

    not_before = _parse_cert_timestamp(peercert.get("notBefore"))
    not_after = _parse_cert_timestamp(peercert.get("notAfter"))
    days_remaining: Optional[int] = None
    if not_after is not None:
        days_remaining = (not_after - datetime.now(timezone.utc)).days

    return CertificateInfo(
        subject_cn=_extract_rdn_field(peercert.get("subject"), "commonName"),
        issuer_cn=_extract_rdn_field(peercert.get("issuer"), "commonName"),
        issuer_org=_extract_rdn_field(peercert.get("issuer"), "organizationName"),
        valid_from=not_before.isoformat() if not_before else None,
        valid_until=not_after.isoformat() if not_after else None,
        days_remaining=days_remaining,
        verified=verified,
    )


class NetworkInspector:
    """Inspeciona protocolos de rede/transporte (DNS, TCP, TLS, ALPN) e respostas HTTP."""

    def __init__(
        self,
        timeout: float = 10.0,
        cache_transport_per_host: bool = True,
    ) -> None:
        self.timeout = timeout
        self.cache_transport_per_host = cache_transport_per_host
        self._transport_cache: Dict[Tuple[str, int, bool], Dict[str, Any]] = {}

    def inspect_transport(
        self, host: str, port: int, use_tls: bool
    ) -> Dict[str, Any]:
        """Resolve DNS, mede handshake TCP/TLS, extrai ALPN, cipher e certificado."""
        cache_key = (host, port, use_tls)
        if self.cache_transport_per_host and cache_key in self._transport_cache:
            return dict(self._transport_cache[cache_key])

        info: Dict[str, Any] = {
            "resolved_ips": [],
            "dns_ms": None,
            "tcp_ms": None,
            "tls_ms": None,
            "alpn_protocol": None,
            "tls_version": None,
            "cipher_suite": None,
            "certificate": None,
            "error": None,
        }

        if not host:
            info["error"] = "Host vazio para inspeção."
            return info

        try:
            t_dns = perf_counter()
            addr_infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            info["dns_ms"] = round((perf_counter() - t_dns) * 1000.0, 2)
            seen_ips: List[str] = []
            for entry in addr_infos:
                ip = str(entry[4][0])
                if ip not in seen_ips:
                    seen_ips.append(ip)
            info["resolved_ips"] = seen_ips
        except OSError as exc:
            info["error"] = f"Falha DNS: {exc}"
            return info

        try:
            t_tcp = perf_counter()
            with socket.create_connection((host, port), timeout=self.timeout) as sock:
                info["tcp_ms"] = round((perf_counter() - t_tcp) * 1000.0, 2)
                if use_tls:
                    self._inspect_tls_socket(sock, host, info)
        except OSError as exc:
            info["error"] = f"Falha conexão TCP/TLS: {exc}"

        if self.cache_transport_per_host and info["error"] is None:
            self._transport_cache[cache_key] = dict(info)
        return info

    def _inspect_tls_socket(
        self, sock: socket.socket, host: str, info: Dict[str, Any]
    ) -> None:
        ctx = ssl.create_default_context()
        ctx.set_alpn_protocols(["h2", "http/1.1"])
        try:
            t_tls = perf_counter()
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                info["tls_ms"] = round((perf_counter() - t_tls) * 1000.0, 2)
                info["alpn_protocol"] = ssock.selected_alpn_protocol()
                info["tls_version"] = ssock.version()
                cipher = ssock.cipher()
                info["cipher_suite"] = cipher[0] if cipher else None
                info["certificate"] = _build_certificate_info(
                    ssock.getpeercert(), verified=True
                )
        except ssl.SSLError as exc:
            info["certificate"] = CertificateInfo(verified=False)
            info["error"] = f"Falha TLS/certificado: {exc}"

    def inspect_response(
        self,
        response: requests.Response,
        request_url: Optional[str] = None,
        total_ms: Optional[float] = None,
        include_transport: bool = True,
    ) -> NetworkTelemetry:
        """Extrai telemetria HTTP da resposta e combina com inspeção DNS/TCP/TLS."""
        final_url = str(getattr(response, "url", "") or request_url or "")
        initial_url = str(request_url or final_url)
        parsed = urlparse(final_url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)

        raw = getattr(response, "raw", None)
        raw_version = getattr(raw, "version", None) if raw is not None else None
        http_version = _format_http_version(raw_version)

        redirect_chain: List[RedirectHop] = []
        for hop in getattr(response, "history", None) or []:
            hop_raw = getattr(hop, "raw", None)
            hop_ver = getattr(hop_raw, "version", None) if hop_raw is not None else None
            redirect_chain.append(
                RedirectHop(
                    url=str(hop.url),
                    status_code=int(hop.status_code),
                    http_version=_format_http_version(hop_ver),
                )
            )

        elapsed = getattr(response, "elapsed", None)
        ttfb_ms = (
            round(elapsed.total_seconds() * 1000.0, 2)
            if elapsed is not None
            else None
        )
        headers = getattr(response, "headers", {}) or {}
        content_type = headers.get("Content-Type") or headers.get("content-type")

        transport: Dict[str, Any] = {}
        if include_transport and host:
            transport = self.inspect_transport(
                host=host, port=port, use_tls=(scheme == "https")
            )

        return NetworkTelemetry(
            url=initial_url,
            final_url=final_url,
            scheme=scheme,
            host=host,
            port=port,
            resolved_ips=transport.get("resolved_ips", []),
            http_version=http_version,
            alpn_protocol=transport.get("alpn_protocol"),
            tls_version=transport.get("tls_version"),
            cipher_suite=transport.get("cipher_suite"),
            certificate=transport.get("certificate"),
            status_code=getattr(response, "status_code", None),
            content_type=content_type,
            redirect_chain=redirect_chain,
            dns_ms=transport.get("dns_ms"),
            tcp_ms=transport.get("tcp_ms"),
            tls_ms=transport.get("tls_ms"),
            ttfb_ms=ttfb_ms,
            total_ms=round(total_ms, 2) if total_ms is not None else ttfb_ms,
            error=transport.get("error"),
        )

    def inspect_url(
        self,
        url: str,
        session: Optional[requests.Session] = None,
        method: str = "GET",
    ) -> NetworkTelemetry:
        """Executa requisição e inspeção completa de URL tratando falhas com segurança."""
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port or (443 if scheme == "https" else 80)

        active_session = session or requests.Session()
        t0 = perf_counter()
        try:
            response = active_session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                allow_redirects=True,
            )
            total_ms = (perf_counter() - t0) * 1000.0
            return self.inspect_response(
                response, request_url=url, total_ms=total_ms, include_transport=True
            )
        except requests.RequestException as exc:
            total_ms = (perf_counter() - t0) * 1000.0
            transport = (
                self.inspect_transport(host=host, port=port, use_tls=(scheme == "https"))
                if host
                else {}
            )
            err_msg = str(exc)
            if transport.get("error"):
                err_msg = f"{err_msg} | {transport['error']}"
            return NetworkTelemetry(
                url=url,
                final_url=url,
                scheme=scheme,
                host=host,
                port=port,
                resolved_ips=transport.get("resolved_ips", []),
                alpn_protocol=transport.get("alpn_protocol"),
                tls_version=transport.get("tls_version"),
                cipher_suite=transport.get("cipher_suite"),
                certificate=transport.get("certificate"),
                dns_ms=transport.get("dns_ms"),
                tcp_ms=transport.get("tcp_ms"),
                tls_ms=transport.get("tls_ms"),
                total_ms=round(total_ms, 2),
                error=err_msg,
            )
