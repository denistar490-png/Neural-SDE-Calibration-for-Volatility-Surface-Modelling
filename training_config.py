"""Per maturity settings loaded from configs/*.json."""
from dataclasses import dataclass

@dataclass(frozen=True)
class AcceptedSliceConfig:
    maturity_index: int
    dtype: str
    training_seed_base: int
    validation_seeds: tuple[int, ...]
    validation_paths_per_seed: int
    path_schedule: tuple[tuple[int, int, int], ...]
    hedge_volatility_mode: str
    numerical_mode: str
    learning_rate: float = 0.001
    maximum_iterations: int = 3_000
    validation_interval: int = 250
    steps_per_day: int = 1

    def paths_for_iteration(self, iteration: int) -> int:
        for first, last, paths in self.path_schedule:
            if first <= iteration <= last:
                return paths
        raise ValueError(f"iteration {iteration} is outside the accepted schedule")

