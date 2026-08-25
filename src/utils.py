from icp import ICPResult


def convergence_ratio(ICP_results: list[ICPResult]) -> float:
    """
    Returns the fraction of runs that have converged to a solution.
    """
    return len([r for r in ICP_results if r.converged == True]) / len(ICP_results)


def convergence_to_global_opt_ratio(ICP_results: list[ICPResult], tol=1e-3) -> float:
    """
    Returns the fraction of runs that have converged to the globally optimal solution. This is measured by small residual errors.
    """
    return len([r for r in ICP_results if r.mean_residuals[-1] < tol]) / len(ICP_results)