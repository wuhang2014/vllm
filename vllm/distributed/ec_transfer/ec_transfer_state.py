# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional

from vllm.distributed.ec_transfer.ec_connector.base import (ECConnectorBase,
                                                            ECConnectorRole)
from vllm.distributed.ec_transfer.ec_connector.factory import (
    ECConnectorFactory)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_EC_CONNECTOR_AGENT: Optional[ECConnectorBase] = None


def get_ec_transfer() -> ECConnectorBase:
    assert _EC_CONNECTOR_AGENT is not None, (
        "disaggregated EC cache is not initialized")
    return _EC_CONNECTOR_AGENT


def has_ec_transfer() -> bool:
    return _EC_CONNECTOR_AGENT is not None


def ensure_ec_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """
    Initialize KV cache transfer parallel group.
    """

    global _EC_CONNECTOR_AGENT

    if vllm_config.ec_transfer_config is None:
        return

    if (vllm_config.ec_transfer_config.is_ec_transfer_instance
            and _EC_CONNECTOR_AGENT is None):
        _EC_CONNECTOR_AGENT = ECConnectorFactory.create_connector(
            config=vllm_config, role=ECConnectorRole.WORKER)
