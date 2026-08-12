from __future__ import annotations

from .profile import StrategyProfile, WeightedSize


def _profile(
    name: str,
    tightness: str,
    *,
    open_frequency: float,
    call: float,
    aggression: float,
    bluff: float,
    sizes: tuple[float, ...] = (0.5, 0.75),
) -> StrategyProfile:
    ranges = {
        "nit": ("TT+,AQs+,AKo", "JJ+,AKs,AKo", "99+,AJs+,AQo+"),
        "tag": ("77+,ATs+,KQs,AJo+", "99+,AJs+,AQo+", "55+,A8s+,KTs+,QJs,ATo+"),
        "lag": (
            "22+,A2s+,K8s+,Q9s+,J9s+,ATo+,KJo+",
            "55+,A8s+,KTs+,QJs,AJo+",
            "22+,A2s+,K5s+,Q8s+,J8s+,T8s+,A8o+,KTo+",
        ),
        "wide": ("22+,A2s+,K2s+,Q5s+,J7s+,T7s+,A2o+,K8o+,Q9o+",) * 3,
    }
    early, default, late = ranges[tightness]
    opens = (
        ("default", default),
        ("UTG", early),
        ("HJ", default),
        ("CO", late),
        ("BTN", late),
        ("BTN/SB", late),
        ("SB", default),
        ("BB", default),
    )
    buckets = ("air", "draw", "weak_made", "medium_made", "strong_made", "monster")
    aggression_values = (
        bluff,
        min(1, aggression * 0.8),
        aggression * 0.4,
        aggression * 0.65,
        min(1, aggression * 1.05),
        min(1, aggression * 1.15),
    )
    call_values = (
        max(0.01, call * 0.1),
        call * 0.65,
        call * 0.5,
        min(1, call * 1.1),
        min(1, call * 0.8),
        min(1, call * 0.5),
    )
    fold_values = tuple(
        max(0, 1 - a - c) for a, c in zip(aggression_values, call_values)
    )
    return StrategyProfile(
        name,
        opens,
        default,
        "QQ+,AKs,AKo",
        "JJ+,AQs+,AKo",
        open_frequency,
        0.05,
        call,
        min(0.95, aggression),
        tuple(zip(buckets, fold_values)),
        tuple(zip(buckets, call_values)),
        tuple(zip(buckets, aggression_values)),
        bluff,
        1.4,
        ((2.5, 0.7), (3.0, 0.3)),
        tuple(WeightedSize(value, 1.0) for value in sizes),
    )


PRESETS = {
    "nit": _profile(
        "Nit", "nit", open_frequency=0.78, call=0.18, aggression=0.45, bluff=0.08
    ),
    "tag": _profile(
        "Tight Aggressive",
        "tag",
        open_frequency=0.88,
        call=0.30,
        aggression=0.62,
        bluff=0.20,
    ),
    "lag": _profile(
        "Loose Aggressive",
        "lag",
        open_frequency=0.93,
        call=0.34,
        aggression=0.76,
        bluff=0.36,
        sizes=(0.33, 0.75, 1.25),
    ),
    "calling_station": _profile(
        "Calling Station",
        "wide",
        open_frequency=0.45,
        call=0.78,
        aggression=0.16,
        bluff=0.04,
        sizes=(0.5,),
    ),
    "maniac": _profile(
        "Maniac",
        "wide",
        open_frequency=0.98,
        call=0.42,
        aggression=0.92,
        bluff=0.62,
        sizes=(0.75, 1.0, 1.5),
    ),
    "bluff_heavy": _profile(
        "Bluff Heavy",
        "lag",
        open_frequency=0.94,
        call=0.25,
        aggression=0.82,
        bluff=0.72,
        sizes=(0.5, 0.75, 1.0),
    ),
}


def preset(name: str) -> StrategyProfile:
    try:
        return PRESETS[name.lower().replace(" ", "_")]
    except KeyError as error:
        raise KeyError(f"unknown synthetic personality {name!r}") from error
