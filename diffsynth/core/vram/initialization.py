from contextlib import contextmanager

import torch


@contextmanager
def skip_model_initialization(device=None):
    if device is None:
        device = torch.device("meta")

    def register_empty_parameter(self, name, param):
        old_register_parameter(self, name, param)
        if param is not None:
            param_cls = type(self._parameters[name])
            kwargs = self._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            self._parameters[name] = param_cls(self._parameters[name].to(device), **kwargs)

    old_register_parameter = torch.nn.Module.register_parameter
    torch.nn.Module.register_parameter = register_empty_parameter
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
