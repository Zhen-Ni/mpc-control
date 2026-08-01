from .plant import Plant, LoggedPlant
from .kalman import Ekf, Ukf
from .rls import Rls
from .discrete import LtiSystem, AtiSystem, HomogeneousSystem, NonlinearSystem
from .mpc import Mpc


def _get_version() -> str:
    """Try to get the installed package version.

    If the package is not installed (e.g., running from source in
 development mode), fall back to "dev".
    """
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("mpc-control")
    except PackageNotFoundError:
        return "dev"


__version__ = _get_version()
