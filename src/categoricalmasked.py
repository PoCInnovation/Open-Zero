import torch as T
from torch.distributions import Categorical
from typing import Optional

class CategoricalMasked(Categorical):

    def __init__(self, logits: T.Tensor, mask: Optional[T.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = T.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)