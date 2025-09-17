from typing import Optional, Dict


class EarlyStopping:
    """Early stops the training if a given metric doesn't improve after a given patience."""

    def __init__(self, monitor_key: str = "loss", patience: int = 7, delta: float = 0,
                 metric_trend: str = "decreasing", trace_func=print):
        """
        Initialize early stopper callback function.

        @param monitor_key: key for the monitor to use withing metric report (dictionary of metrics, including loss)
        @param patience: how many epochs to continue non-improving training epochs
        @param delta: margin of error to consider.  0 will look at exact values, any value added will make the early
                      stopper more lenient
        @param metric_trend: "decreasing" or "increasing", depending on metric (e.g., loss should be decreasing, F1
                             should be increasing)
        @param trace_func: output trace function, defaults at print on console
        """
        self.monitor_key: str = monitor_key
        self.patience: int = patience
        self.counter: int = 0
        self.delta: float = delta
        self.best_score: Optional[float] = None
        self.exit_flag: bool = False
        self.mode: str = metric_trend
        self.trace_func = trace_func

    def __call__(self, metric_report: Dict[str, float]):
        """
        Call early stopper.

        @param metric_report: Dictionary of values for metrics, key (str) : value (float)
        @return: True if metric has improved. False if it has not.
        """
        score: float = metric_report[self.monitor_key]
        # Init
        if self.best_score is None:
            self.best_score = score
            return True
        # Incr / Decr trend
        elif (self.mode == "increasing" and score < self.best_score - self.delta) or \
                (self.mode == "decreasing" and score > self.best_score + self.delta):
            self.counter += 1
            # Logs
            self.trace_func(f"- EarlyStopping || Mode - {self.mode} || {self.monitor_key} value {score:.4f} "
                            f" is {'<' if self.mode == 'increasing' else '>'} "
                            f"than {(self.best_score + (self.delta if self.mode == 'decreasing' else -self.delta)):.4f}"
                            f" (best score + delta)")
            self.trace_func(f"- EarlyStopping || Counter    || {self.counter} out of {self.patience}")
            # Exit con
            if self.counter >= self.patience:
                # Will make the training stop
                self.exit_flag = True
                self.trace_func(f"- EarlyStopping || Stopping early...")
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True
