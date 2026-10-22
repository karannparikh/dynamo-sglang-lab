"""``resolved_view`` -- the effective configuration projection.

The fake ServerArgs is already its own resolved view.
"""

from typing import Any


def resolved_view(server_args: Any) -> Any:
    return server_args
