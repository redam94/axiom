"""ArviZ interop: ``Posterior`` <-> ``InferenceData`` / ``DataTree``, lazily.

arviz is an optional extra in two generations: 0.x (``arviz.from_dict`` ->
``InferenceData``) and 1.x (``arviz_base.from_dict`` -> an xarray
``DataTree``). ``to_inference_data`` tries ``arviz_base`` first, then
``arviz``, and returns ``Unsupported`` when neither imports. Nothing here is
imported at module top, so ``import axiom.infer`` stays light (gate 1).

Provenance travels two ways: the whole dict as one ``provenance_json`` attr
(the lossless round trip — bools stay bools, ``None`` stays ``None``, numpy
scalars become Python ones) and flattened per-key attrs for humans reading
the file.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from axiom.core.posterior import Posterior
from axiom.core.result import Unsupported

__all__ = ["from_inference_data", "to_inference_data"]

_SAMPLE_DIMS = ("chain", "draw")
_PROVENANCE_ATTR = "provenance_json"


def _dims(posterior: Posterior) -> dict[str, list[str]]:
    """Trailing-axis dimension names per variable, from the coords convention ``name_dim{i}``."""
    out: dict[str, list[str]] = {}
    for name in sorted(posterior.names()):
        a = posterior.draws(name)
        if a.ndim > 2:
            out[name] = [f"{name}_dim{i}" for i in range(a.ndim - 2)]
    return out


def _json_default(v: Any) -> Any:
    """numpy scalars and arrays become Python values; anything else becomes its ``str``."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _json_attrs(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance as netCDF-friendly attrs: one lossless JSON blob plus flattened keys."""
    out: dict[str, Any] = {
        _PROVENANCE_ATTR: json.dumps(provenance, sort_keys=True, default=_json_default)
    }
    for k, v in provenance.items():
        if k == _PROVENANCE_ATTR:
            continue
        v = _python_scalar(v)
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, int | float | str):
            out[k] = v
        elif v is None:
            out[k] = "null"
        else:
            out[k] = json.dumps(v, sort_keys=True, default=_json_default)
    return out


def to_inference_data(posterior: Posterior) -> Any | Unsupported:
    """Build an arviz object with a ``posterior`` group; provenance goes into its attrs.

    Returns a ``DataTree`` under arviz 1.x (``arviz_base``), an
    ``InferenceData`` under arviz 0.x, or ``Unsupported`` if neither is
    installed.
    """
    draws = {name: posterior.draws(name) for name in sorted(posterior.names())}
    dims = _dims(posterior)
    coords = {k: list(v) for k, v in posterior.coords().items()}
    attrs = _json_attrs(posterior.provenance)
    from_dict_1x = _import_from_dict("arviz_base")
    if from_dict_1x is not None:
        return from_dict_1x(
            {"posterior": draws},
            dims=dims,
            coords=coords,
            attrs={"posterior": attrs},
        )
    from_dict_0x = _import_from_dict("arviz")
    if from_dict_0x is None:
        return Unsupported(
            reason="arviz is not installed; install axiom[numpyro] for arviz-base",
            missing=("arviz_base", "arviz"),
        )
    return from_dict_0x(posterior=draws, dims=dims, coords=coords, attrs=attrs)


def _import_from_dict(module: str) -> Callable[..., Any] | None:
    """``module.from_dict`` if the module imports, else ``None``."""
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    fn: Callable[..., Any] = mod.from_dict
    return fn


def _posterior_dataset(idata: Any) -> Any:
    """The ``posterior`` group as an xarray ``Dataset`` from either arviz generation."""
    if hasattr(idata, "children"):
        # xarray DataTree (arviz 1.x): groups are children; ``to_dataset`` flattens one.
        node = idata["posterior"]
        return node.to_dataset() if hasattr(node, "to_dataset") else node
    if hasattr(idata, "data_vars"):
        return idata  # already a Dataset
    if isinstance(idata, Mapping):
        return idata["posterior"]
    group = getattr(idata, "posterior", None)
    if group is None:
        raise KeyError("object has no 'posterior' group")
    return group.to_dataset() if hasattr(group, "to_dataset") else group


def _python_scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _decode_attr(v: Any) -> Any:
    """Best-effort decoding of a flattened attr (files written without ``provenance_json``)."""
    v = _python_scalar(v)
    if isinstance(v, str):
        if v == "null":
            return None
        if v[:1] in "[{":
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
    return v


_ARVIZ_BOOKKEEPING = frozenset(
    {
        "created_at",
        "creation_library",
        "creation_library_version",
        "creation_library_language",
        "sample_dims",
        "arviz_version",
        "inference_library",
        "inference_library_version",
    }
)


def _provenance_from_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    blob = attrs.get(_PROVENANCE_ATTR)
    if isinstance(blob, str):
        decoded = json.loads(blob)
        if isinstance(decoded, dict):
            return {str(k): v for k, v in decoded.items()}
    provenance = {
        str(k): _decode_attr(v)
        for k, v in attrs.items()
        if k not in _ARVIZ_BOOKKEEPING and k != _PROVENANCE_ATTR
    }
    for k in ("hessian_pd", "converged", "verified", "optimizer_success"):
        if k in provenance and isinstance(provenance[k], int):
            provenance[k] = bool(provenance[k])
    return provenance


def from_inference_data(idata: Any) -> Posterior:
    """Read the ``posterior`` group of an arviz object back into a ``Posterior``.

    Variables keep their ``(chain, draw, *shape)`` layout; named trailing
    dimensions become coords; the group's ``provenance_json`` attr becomes
    provenance verbatim. Objects written without it (another tool, an older
    file) fall back to decoding the flattened attrs, with arviz's own
    bookkeeping attrs dropped.
    """
    ds = _posterior_dataset(idata)
    draws: dict[str, np.ndarray[Any, Any]] = {}
    coords: dict[str, list[Any]] = {}
    for name, da in ds.data_vars.items():
        dims = tuple(da.dims)
        if dims[:2] != _SAMPLE_DIMS:
            da = da.transpose(*_SAMPLE_DIMS, *[d for d in dims if d not in _SAMPLE_DIMS])
            dims = tuple(da.dims)
        draws[str(name)] = np.asarray(da.values, dtype=float)
        for d in dims[2:]:
            if d in ds.coords:
                coords[str(d)] = [_python_scalar(x) for x in ds.coords[d].values]
    return Posterior(draws, coords=coords, provenance=_provenance_from_attrs(ds.attrs))
