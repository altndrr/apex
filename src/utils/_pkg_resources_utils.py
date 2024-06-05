import pkg_resources

__all__ = ["package_available"]


def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    Args:
    ----
        package_name (str): Name of the package to check.

    """
    try:
        return pkg_resources.require(package_name) is not None
    except pkg_resources.DistributionNotFound:
        return False
