from torch.utils.data.sampler import BatchSampler, Sampler
import itertools
from typing import Iterator, Union, Iterable
from torch.utils.data import Sampler

class CustomBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler_iter = iter(self.sampler)

    def __iter__(self) -> Iterator[list[int]]:
        
        while True:
            batch = []
            while len(batch) < self.batch_size:
                try:
                    batch.append(next(self.sampler_iter))
                except StopIteration:
                    self.sampler_iter = iter(self.sampler)
                    if not batch:
                        continue
                    else:
                        if not self.drop_last:
                            yield batch
                        return
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]