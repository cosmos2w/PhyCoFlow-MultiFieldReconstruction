"""Compatibility imports for project flow source priors."""

from .pointcloud.core.flow_priors import IIDGaussianPrior, RFFGaussianPrior

__all__ = ["IIDGaussianPrior", "RFFGaussianPrior"]
