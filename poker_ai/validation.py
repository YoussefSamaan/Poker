"""Small deterministic exact-versus-sampling development checks."""

from __future__ import annotations

from .multiway import MultiwayEquityCalculator
from .ranges import WeightedRange


def multiway_validation_rows(samples: int = 5_000) -> tuple[dict[str, float], ...]:
    ranges = (
        WeightedRange.from_mapping({"AcAd": 2, "KcKd": 1}),
        WeightedRange.from_mapping({"AcQh": 3, "JhTs": 1}),
        WeightedRange.from_mapping({"KsQd": 1, "7h7d": 2}),
    )
    calculator = MultiwayEquityCalculator()
    rows = []
    for opponents in (2, 3):
        exact = calculator.calculate(
            "2c 3d", "4h 5s 9c", ranges[:opponents], exact=True
        )
        monte_carlo = calculator.calculate(
            "2c 3d",
            "4h 5s 9c",
            ranges[:opponents],
            exact=False,
            samples=samples,
            seed=100 + opponents,
        )
        rows.append(
            {
                "opponents": float(opponents),
                "exact_equity": exact.equity,
                "monte_carlo_equity": monte_carlo.equity,
                "absolute_error": abs(exact.equity - monte_carlo.equity),
                "monte_carlo_standard_error": monte_carlo.standard_error,
            }
        )
    return tuple(rows)


def main() -> None:
    for row in multiway_validation_rows():
        print(
            f"opponents={int(row['opponents'])} "
            f"exact={row['exact_equity']:.5f} "
            f"mc={row['monte_carlo_equity']:.5f} "
            f"abs_error={row['absolute_error']:.5f} "
            f"mc_se={row['monte_carlo_standard_error']:.5f}"
        )


if __name__ == "__main__":
    main()
