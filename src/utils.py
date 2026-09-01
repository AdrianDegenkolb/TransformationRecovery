from icp import ICPResult, MultiStartICPResult


def convergence_ratio(ICP_results: list[ICPResult | MultiStartICPResult]) -> float:
    """
    Returns the fraction of runs that have converged to a solution.
    """
    return sum(r.converged for r in ICP_results) / len(ICP_results)


def convergence_to_global_opt_ratio(ICP_results: list[ICPResult | MultiStartICPResult], tol: float = 1e-3) -> float:
    """
    Returns the fraction of runs that have converged to the globally optimal solution. This is measured by small residual errors.
    """
    return sum(r.mean_residuals[-1] < tol for r in ICP_results) / len(ICP_results)
