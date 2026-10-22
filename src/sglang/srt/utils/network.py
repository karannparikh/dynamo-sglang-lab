"""Address helpers.

``NetworkAddress`` is the surface Dynamo's ``_disagg.compute_bootstrap_address``
and ``BaseWorkerHandler._get_bootstrap_info`` depend on. Both call
``to_host_port_str().rsplit(":", 1)[0]`` to recover a bare host, so IPv6 must
come back bracketed or that split would cut the address in half.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Optional


class NetworkAddress:
    def __init__(self, host: str, port: Optional[int] = None) -> None:
        self.host = str(host).strip().strip("[]")
        self.port = int(port) if port is not None else None

    @property
    def is_ipv6(self) -> bool:
        try:
            return isinstance(ipaddress.ip_address(self.host), ipaddress.IPv6Address)
        except ValueError:
            return False

    @classmethod
    def parse(cls, addr: str) -> "NetworkAddress":
        text = str(addr).strip()
        if text.startswith("["):                       # [::1]:8998
            host, _, rest = text[1:].partition("]")
            port = rest.lstrip(":") or None
            return cls(host, int(port) if port else None)
        if text.count(":") > 1:                        # bare IPv6, no port
            return cls(text, None)
        host, _, port = text.partition(":")
        return cls(host, int(port) if port else None)

    def resolved(self) -> "NetworkAddress":
        """Resolve a hostname to an address; an IP literal resolves to itself."""
        try:
            ipaddress.ip_address(self.host)
            return self
        except ValueError:
            pass
        try:
            info = socket.getaddrinfo(self.host, self.port, proto=socket.IPPROTO_TCP)
            return NetworkAddress(info[0][4][0], self.port)
        except socket.gaierror:
            return self

    def to_host_port_str(self) -> str:
        host = f"[{self.host}]" if self.is_ipv6 else self.host
        return f"{host}:{self.port}" if self.port is not None else host

    def to_tcp(self) -> str:
        return f"tcp://{self.to_host_port_str()}"

    def __repr__(self) -> str:
        return f"NetworkAddress({self.to_host_port_str()})"


def get_local_ip_auto() -> str:
    """SGLANG_HOST_IP wins; otherwise probe, then fall back to loopback.

    The fallback matters here: on a laptop with no route the probe fails, and a
    single-box prefill/decode pair reaches each other over loopback anyway.
    """
    override = os.environ.get("SGLANG_HOST_IP") or os.environ.get("HOST_IP")
    if override:
        return override.strip().strip("[]")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))       # no packets sent; just picks a route
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def get_zmq_socket(context, socket_type, endpoint: str, bind: bool):
    socket_ = context.socket(socket_type)
    if bind:
        socket_.bind(endpoint)
    else:
        socket_.connect(endpoint)
    return socket_
