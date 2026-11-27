"""Módulo de inspeção de protocolos de rede, TLS/HTTP e captura opcional."""

from src.network.inspector import NetworkInspector
from src.network.models import (
    CertificateInfo,
    NetworkTelemetry,
    RedirectHop,
    aggregate_protocol_telemetry,
)
from src.network.packet_capture import (
    PacketCaptureSummary,
    capture_network_traffic,
    detect_capture_backend,
    parse_tshark_protocol_output,
)

__all__ = [
    "CertificateInfo",
    "NetworkInspector",
    "NetworkTelemetry",
    "PacketCaptureSummary",
    "RedirectHop",
    "aggregate_protocol_telemetry",
    "capture_network_traffic",
    "detect_capture_backend",
    "parse_tshark_protocol_output",
]
