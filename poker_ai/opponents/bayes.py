from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class BetaEstimate:
    """Transparent Beta-Binomial estimate with an exact numerical interval."""

    successes: int = 0
    failures: int = 0
    alpha_prior: float = 1.0
    beta_prior: float = 1.0

    def __post_init__(self) -> None:
        if self.successes < 0 or self.failures < 0:
            raise ValueError("Beta counts must be non-negative")
        if self.alpha_prior <= 0 or self.beta_prior <= 0:
            raise ValueError("Beta prior parameters must be positive")

    @property
    def opportunities(self) -> int:
        return self.successes + self.failures

    @property
    def alpha(self) -> float:
        return self.alpha_prior + self.successes

    @property
    def beta(self) -> float:
        return self.beta_prior + self.failures

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total * total * (total + 1))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        if not 0 < level < 1:
            raise ValueError("credible interval level must be between zero and one")
        tail = (1 - level) / 2
        return (
            _beta_quantile(tail, self.alpha, self.beta),
            _beta_quantile(1 - tail, self.alpha, self.beta),
        )


def _beta_quantile(probability: float, alpha: float, beta: float) -> float:
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(70):
        middle = (low + high) / 2
        if _regularized_beta(middle, alpha, beta) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _regularized_beta(value: float, alpha: float, beta: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    factor = math.exp(
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + alpha * math.log(value)
        + beta * math.log1p(-value)
    )
    if value < (alpha + 1) / (alpha + beta + 2):
        return factor * _beta_fraction(value, alpha, beta) / alpha
    return 1 - factor * _beta_fraction(1 - value, beta, alpha) / beta


def _beta_fraction(value: float, alpha: float, beta: float) -> float:
    tiny = 1e-300
    qab, qap, qam = alpha + beta, alpha + 1, alpha - 1
    c = 1.0
    d = 1 - qab * value / qap
    d = 1 / max(abs(d), tiny) * (1 if d >= 0 else -1)
    result = d
    for index in range(1, 201):
        even = 2 * index
        coefficient = index * (beta - index) * value / (
            (qam + even) * (alpha + even)
        )
        d = 1 + coefficient * d
        d = d if abs(d) > tiny else tiny
        c = 1 + coefficient / c
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        result *= d * c
        coefficient = -(alpha + index) * (qab + index) * value / (
            (alpha + even) * (qap + even)
        )
        d = 1 + coefficient * d
        d = d if abs(d) > tiny else tiny
        c = 1 + coefficient / c
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        delta = d * c
        result *= delta
        if abs(delta - 1) < 3e-14:
            break
    return result
