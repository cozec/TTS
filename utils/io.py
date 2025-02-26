from typing import Any, Union, Callable, Dict
import fsspec
import torch

def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    if cache:
        # ... existing cache logic ...
    else:
        with fsspec.open(path, "rb") as f:
            # Explicitly set weights_only=False
            return torch.load(f, map_location=map_location, weights_only=False, **kwargs) 