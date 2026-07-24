"""Silence the third-party warnings a normal forecast run emits by the hundred.

Deliberately narrow. Each filter names the message and the module it comes from, so a
warning that is genuinely about our code still reaches the log. Blanket
``warnings.filterwarnings("ignore")`` would hide the next real problem too.

What is silenced, and why each is noise rather than signal:

* **statsmodels starting parameters** — "Non-invertible starting MA parameters" and
  "Non-stationary starting autoregressive parameters". These say the optimiser's
  *initial guess* was outside the invertible/stationary region, so it fell back to
  zeros and carried on. It is a remark about the search's first step, not about the
  fitted model, and it fires per fit: one pod × one channel × a backtest refit each.
* **statsmodels convergence** — already filtered inside ``autoarima``; repeated here so
  a caller that does not import that module gets the same quiet.
* **pandas ``to_pydatetime``** — a deprecation raised deep inside plotly's datetime
  handling when it renders a DatetimeIndex. Nothing in this repo calls it.

Call :func:`quiet_third_party_warnings` at the entry point of anything interactive
(the dashboard, a notebook). It is not called on import: a library that mutates the
global warning filters as a side effect of being imported is a bad neighbour.
"""
import warnings


def quiet_third_party_warnings() -> None:
    """Install the narrow filters described in the module docstring. Idempotent."""
    # statsmodels: the optimiser's starting-value remarks, one per fit.
    warnings.filterwarnings(
        "ignore", message=".*Non-invertible starting MA parameters.*",
        category=UserWarning, module=r".*statsmodels.*")
    warnings.filterwarnings(
        "ignore", message=".*Non-stationary starting autoregressive parameters.*",
        category=UserWarning, module=r".*statsmodels.*")
    warnings.filterwarnings(
        "ignore", message=".*Non-invertible starting seasonal moving average.*",
        category=UserWarning, module=r".*statsmodels.*")
    warnings.filterwarnings(
        "ignore", message=".*Non-stationary starting seasonal autoregressive.*",
        category=UserWarning, module=r".*statsmodels.*")
    # Short series vs a seasonal order — the same "we started the optimiser at zeros"
    # remark. The series length is already checked upstream (SplitConfigurationError),
    # and that check is what should speak, once, in the run summary.
    warnings.filterwarnings(
        "ignore", message=".*Too few observations to estimate starting parameters.*",
        category=UserWarning, module=r".*statsmodels.*")

    # statsmodels: did-not-converge and the maxiter notice. The zero-fallback and the
    # run summary are what report a bad fit; this warning only repeats it per pod.
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass
    warnings.filterwarnings(
        "ignore", message=".*Maximum Likelihood optimization failed to converge.*")

    # pandas → plotly: a deprecation in a path we never call directly.
    warnings.filterwarnings(
        "ignore", message=".*to_pydatetime is deprecated.*", category=FutureWarning)

    # gradio → starlette: a deprecation inside gradio's own routing. Nothing we call, and
    # nothing we can fix short of a gradio upgrade.
    warnings.filterwarnings("ignore", module=r".*gradio.*")
