"""Modo opcional de captura/análise de pacotes em interface local (tshark ou scapy)."""

from collections import Counter
from dataclasses import asdict, dataclass, field
import importlib.util
import shutil
import subprocess
from typing import Any, Dict, Iterable, Optional

_PROTOCOL_ALIASES = {
    "tcp": "TCP",
    "udp": "UDP",
    "dns": "DNS",
    "tls": "TLS",
    "ssl": "TLS",
    "http": "HTTP",
    "http2": "HTTP/2",
    "quic": "QUIC",
    "icmp": "ICMP",
}


@dataclass
class PacketCaptureSummary:
    """Resumo de protocolos identificados em captura controlada de tráfego."""

    available: bool
    backend: Optional[str] = None
    interface: Optional[str] = None
    duration_seconds: int = 5
    total_packets: int = 0
    protocol_counts: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_capture_backend() -> Optional[str]:
    """Detecta se 'tshark' (CLI) ou 'scapy' (Python) estão disponíveis no ambiente."""
    if shutil.which("tshark") is not None:
        return "tshark"
    if importlib.util.find_spec("scapy") is not None:
        return "scapy"
    return None


def parse_tshark_protocol_output(lines: Iterable[str]) -> Dict[str, int]:
    """Conta ocorrências de protocolos (TCP, UDP, DNS, TLS, HTTP...) a partir de frame.protocols."""
    counts: Counter = Counter()
    for line in lines:
        tokens = [tok.strip().lower() for tok in line.split(":") if tok.strip()]
        seen_in_packet = set()
        for tok in tokens:
            mapped = _PROTOCOL_ALIASES.get(tok)
            if mapped and mapped not in seen_in_packet:
                seen_in_packet.add(mapped)
                counts[mapped] += 1
    return dict(counts)


def capture_network_traffic(
    interface: Optional[str] = None,
    duration_seconds: int = 5,
    packet_count: int = 100,
) -> PacketCaptureSummary:
    """Executa captura opcional se tshark ou scapy estiver instalado; falha graciosamente caso contrário."""
    backend = detect_capture_backend()
    if backend is None:
        return PacketCaptureSummary(
            available=False,
            backend=None,
            interface=interface,
            duration_seconds=duration_seconds,
            error="Nenhum backend de captura (tshark ou scapy) disponível no ambiente.",
        )

    if backend == "tshark":
        cmd = [
            "tshark",
            "-a",
            f"duration:{duration_seconds}",
            "-c",
            str(packet_count),
            "-T",
            "fields",
            "-e",
            "frame.protocols",
        ]
        if interface:
            cmd.extend(["-i", interface])
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration_seconds + 10,
                check=True,
            )
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            return PacketCaptureSummary(
                available=True,
                backend="tshark",
                interface=interface,
                duration_seconds=duration_seconds,
                total_packets=len(lines),
                protocol_counts=parse_tshark_protocol_output(lines),
            )
        except Exception as exc:
            return PacketCaptureSummary(
                available=True,
                backend="tshark",
                interface=interface,
                duration_seconds=duration_seconds,
                error=f"Falha ao executar tshark: {exc}",
            )

    try:
        from scapy.all import DNS, TCP, UDP, sniff  # type: ignore

        packets = sniff(
            iface=interface,
            count=packet_count,
            timeout=duration_seconds,
        )
        counts: Counter = Counter()
        for pkt in packets:
            if pkt.haslayer(TCP):
                counts["TCP"] += 1
            if pkt.haslayer(UDP):
                counts["UDP"] += 1
            if pkt.haslayer(DNS):
                counts["DNS"] += 1
        return PacketCaptureSummary(
            available=True,
            backend="scapy",
            interface=interface,
            duration_seconds=duration_seconds,
            total_packets=len(packets),
            protocol_counts=dict(counts),
        )
    except Exception as exc:
        return PacketCaptureSummary(
            available=True,
            backend="scapy",
            interface=interface,
            duration_seconds=duration_seconds,
            error=f"Falha ao capturar via scapy: {exc}",
        )
