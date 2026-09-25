from __future__ import annotations

import inspect
from typing import Any, Dict, Type


def supports_parameter(cls_or_fn: Any, name: str) -> bool:
    signature = inspect.signature(cls_or_fn.__init__ if inspect.isclass(cls_or_fn) else cls_or_fn)
    return name in signature.parameters


def build_config(config_cls: Type[Any], raw_kwargs: Dict[str, Any]) -> Any:
    signature = inspect.signature(config_cls.__init__)
    params = signature.parameters
    kwargs = dict(raw_kwargs)

    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return config_cls(**kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in params}
    return config_cls(**filtered)


def trainer_tokenizer_kwargs(trainer_cls: Type[Any], tokenizer: Any) -> Dict[str, Any]:
    signature = inspect.signature(trainer_cls.__init__)
    params = signature.parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    return {}

