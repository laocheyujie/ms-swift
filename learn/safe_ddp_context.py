import torch.distributed as dist
from contextlib import contextmanager
from swift.utils.env import is_dist, is_master, is_local_master


@contextmanager
def safe_ddp_context():
        if is_dist():
            if not is_master():
                dist.barrier()
            if not is_local_master():
                # Compatible with multi-machine scenarios,
                # where each machine uses different storage hardware.
                dist.barrier()
        yield
        if is_dist():
            if is_master():
                dist.barrier()
            if is_local_master():
                dist.barrier()
