# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union
import torch
from dataclasses import dataclass
import threading
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class BufferSegment:
    """Represents a segment in the ring buffer."""
    start_offset: int
    size: int
    tensor: Optional[torch.Tensor] = None
    is_allocated: bool = False


class ECRingBuffer:
    """
    A ring buffer for managing GPU memory allocation and deallocation of torch tensors.
    
    This class implements a ring buffer algorithm to efficiently manage a pre-allocated
    buffer on a specified device. It provides methods to allocate and deallocate tensor
    memory at runtime while avoiding frequent CUDA memory allocation/deallocation calls.
    
    Key Features:
    - Pre-allocates a fixed-size buffer on the specified device
    - Uses ring buffer algorithm for efficient memory reuse
    - Thread-safe allocation and deallocation
    - Automatic garbage collection of deallocated segments
    - Support for various tensor dtypes and shapes
    
    Args:
        device: The device to allocate the buffer on (e.g., 'cuda:0', 'cpu')
        buffer_size: Size of the buffer in bytes
        dtype: Default data type for allocations (default: torch.uint8)
    
    Example:
        >>> ring_buffer = ECRingBuffer('cuda:0', 1024 * 1024 * 1024)  # 1GB buffer
        >>> tensor = ring_buffer.allocate_tensor([100, 100], torch.float32)
        >>> ring_buffer.deallocate_tensor(tensor)
    """
    
    def __init__(self, 
                 device: Union[str, torch.device], 
                 buffer_size: int,
                 dtype: torch.dtype = torch.uint8):
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.head = 0  # Current allocation position
        self.tail = 0  # Position where oldest allocation starts
        self.allocated_segments = {}  # Maps tensor data_ptr to BufferSegment
        self._lock = threading.RLock()  # Thread-safe operations
        
        # Pre-allocate the main buffer
        self._initialize_buffer()
        
        logger.info(f"ECRingBuffer initialized on {self.device} with {buffer_size} bytes")
    
    def _initialize_buffer(self):
        """Initialize the pre-allocated buffer on the device."""
        try:
            # Allocate buffer as uint8 tensor for byte-level access
            self.buffer = torch.empty(
                self.buffer_size, 
                dtype=torch.uint8, 
                device=self.device
            )
            self.buffer_ptr = self.buffer.data_ptr()
            logger.debug(f"Buffer allocated at address: {self.buffer_ptr}")
        except Exception as e:
            logger.error(f"Failed to allocate buffer on {self.device}: {e}")
            raise RuntimeError(f"Cannot allocate buffer on {self.device}") from e
    
    def _get_aligned_size(self, size: int, alignment: int = 256) -> int:
        """Get aligned size for memory allocation."""
        return ((size + alignment - 1) // alignment) * alignment
    
    def _find_free_space(self, required_size: int) -> Optional[int]:
        """
        Find a free space in the ring buffer for the required size.
        
        Args:
            required_size: Size in bytes needed
            
        Returns:
            Starting offset if space found, None otherwise
        """
        aligned_size = self._get_aligned_size(required_size)
        
        # Simple ring buffer allocation strategy
        # Try to allocate from current head position
        if self.head + aligned_size <= self.buffer_size:
            # Space available from head to end of buffer
            start_offset = self.head
            return start_offset
        elif aligned_size <= self.tail:
            # Wrap around: space available from beginning to tail
            start_offset = 0
            return start_offset
        else:
            # No space available
            return None
    
    def _cleanup_old_segments(self):
        """Clean up deallocated segments to make room for new allocations."""
        # Remove segments that are no longer allocated
        to_remove = []
        for data_ptr, segment in self.allocated_segments.items():
            if not segment.is_allocated:
                to_remove.append(data_ptr)
        
        for data_ptr in to_remove:
            del self.allocated_segments[data_ptr]
    
    def allocate_tensor(self, 
                       shape: tuple, 
                       dtype: torch.dtype) -> Optional[torch.Tensor]:
        """
        Allocate a tensor from the ring buffer.
        
        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            
        Returns:
            Allocated tensor or None if allocation fails
        """
        with self._lock:
            # Calculate required size
            element_size = torch.tensor(0, dtype=dtype).element_size()
            numel = 1
            for dim in shape:
                numel *= dim
            required_size = numel * element_size
            
            # Clean up old segments first
            self._cleanup_old_segments()
            
            # Find free space
            start_offset = self._find_free_space(required_size)
            if start_offset is None:
                logger.warning(f"Cannot allocate {required_size} bytes, buffer full")
                return None
            
            # Calculate aligned size
            aligned_size = self._get_aligned_size(required_size)
            
            try:
                # Create tensor view from buffer
                buffer_slice = self.buffer[start_offset:start_offset + required_size]
                tensor = buffer_slice.view(dtype).reshape(shape)
                
                # Create segment record
                segment = BufferSegment(
                    start_offset=start_offset,
                    size=aligned_size,
                    tensor=tensor,
                    is_allocated=True
                )
                
                # Update head position
                if start_offset + aligned_size >= self.buffer_size:
                    # Wrapped around
                    self.head = 0
                else:
                    self.head = start_offset + aligned_size
                
                # Record the allocation
                self.allocated_segments[tensor.data_ptr()] = segment
                
                logger.debug(f"Allocated tensor {shape} {dtype} at offset {start_offset}")
                return tensor
                
            except Exception as e:
                logger.error(f"Failed to create tensor view: {e}")
                return None
    
    def deallocate_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Deallocate a tensor from the ring buffer.
        
        Args:
            tensor: Tensor to deallocate
            
        Returns:
            True if deallocation successful, False otherwise
        """
        with self._lock:
            data_ptr = tensor.data_ptr()
            
            if data_ptr not in self.allocated_segments:
                logger.warning("Tensor not found in allocated segments")
                return False
            
            segment = self.allocated_segments[data_ptr]
            segment.is_allocated = False
            segment.tensor = None
            
            logger.debug(f"Deallocated tensor at offset {segment.start_offset}")
            return True
    
    def get_memory_usage(self) -> dict:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        with self._lock:
            allocated_size = sum(
                segment.size for segment in self.allocated_segments.values() 
                if segment.is_allocated
            )
            
            free_size = self.buffer_size - allocated_size
            
            return {
                'total_size': self.buffer_size,
                'allocated_size': allocated_size,
                'free_size': free_size,
                'utilization': allocated_size / self.buffer_size * 100,
                'active_segments': len([s for s in self.allocated_segments.values() if s.is_allocated])
            }
    
    def clear(self):
        """Clear all allocations and reset the buffer."""
        with self._lock:
            self.allocated_segments.clear()
            self.head = 0
            self.tail = 0
            logger.info("Ring buffer cleared")
    
    def __del__(self):
        """Cleanup when the ring buffer is destroyed."""
        if hasattr(self, 'buffer'):
            self.clear()
            logger.debug("ECRingBuffer destroyed")
