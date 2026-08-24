"""Named public exceptions for semantic back-door task APIs."""


class OutOfSupportError(ValueError):
    """Raised when a well-shaped query is outside covariate support."""
