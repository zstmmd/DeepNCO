__all__ = ["ResourceTimeALNSEngine"]


def __getattr__(name):
    if name == "ResourceTimeALNSEngine":
        from .engine import ResourceTimeALNSEngine

        return ResourceTimeALNSEngine
    raise AttributeError(name)
