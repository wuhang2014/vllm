# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ECConnectorBase Class for Distributed Encoder Cache & P2P Encoder cache
communication in V1

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to load/save Encoder cache.
        check_caches_exist() - Check whether Encoder cache of requests
        exist
        update_state_after_alloc() - update ECConnector state after
        allocate. This will decide to load the cache or not 
        request_finished() - called when a request is finished, free the
        cache with the requests

    Worker-side: runs in each worker, loads/saves Encoder Cache to/from
    the Connector based on the metadata.
        start_load_ec() - starts loading all ECs (maybe async)
        wait_for_save() - blocks until all saves are done

        get_finished() - called with ids of finished requests, returns
            ids of requests that have completed async sending/recving.
"""

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput, KVConnectorOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


class ECConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate between the
    Scheduler ECConnector and Worker ECConnector.
    """
    pass


class ECConnectorBase(ABC):

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        self._connector_metadata: Optional[ECConnectorMetadata] = None
        self._vllm_config = vllm_config
        self._role = role
        self._is_producer = (
            vllm_config.ec_transfer_config.ec_role == 'ec_producer')

    @property
    def role(self) -> ECConnectorRole:
        return self._role

    @property
    def is_producer(self) -> bool:
        return self._is_producer

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(
            self, connector_metadata: ECConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time 
        before the model execution. The metadata will be used for runtime
        EC cache loading.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time 
        after the model execution.
        """
        self._connector_metadata = None

    def _get_connector_metadata(self) -> ECConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """

        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def register_cache(
        self,
        ec_cache: torch.Tensor,
    ):
        """
        Initialize with the EC cache.
        Args: 
            ec_cache: tensor of encoder cache
        """
        # TODO: Implement this later for P2P feature
        return

    @abstractmethod
    def start_load_caches(self, **kwargs) -> None:
        """
        Start loading the cache from the connector to vLLM's encoder cache.
        This is called before _gather_mm_embeddings for EC Connector
        and before execute_model for KV Connector
        For EC the encoder_cache and mm_hash is store in kwargs

        Args:
            **kwargs: additional arguments for the load operation
            
        """
        pass

    @abstractmethod
    def save_caches(self, **kwargs) -> None:
        """
        Save caches into connector
        For EC the encoder_cache and mm_hash is store in kwargs
        """
        pass

    @abstractmethod
    def wait_for_save(self):
        """
        Block until all the save operations is done. 
        """
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return None, None

    # ==============================
    # Scheduler-side methods
    # ==============================

    @abstractmethod
    def check_caches_exist(
        self,
        request: "Request",
    ) -> list[bool]:
        """
        Check if encoder cache exists for each mm data of requests

        Args:
            request (Request): the request object.

        Returns:
            A list bool where ith value is True if cache exists for
            ith mm_data of requests
        """
        pass

    @abstractmethod
    def update_state_after_alloc(self, request: "Request", index: int):
        """
        Update ECConnector state to decide allocate cache for requests

        Args:
            request (Request): the request object.
        """
        pass

    @abstractmethod
    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> ECConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        return

    def request_finished(
            self, request: "Request", ec_connector_output: ECConnectorOutput,
        ) -> Optional[dict[str, Any]]:
        """
        Called when a request has finished, before its freed the local
        encoder cached.

        Returns:
            True if the request is being saved/sent asynchronously and
            cached should not be freed until the request_id is returned
            from get_finished().
        """
        return None
