# Learned Opponent Model v2

This milestone introduces the first supervised CPU models while preserving the
statistical model as the production research baseline. The learned package is
strictly offline and never enters `HoldemGame`.

The project declares `scikit-learn>=1.5,<2`, `numpy>=1.24,<3`, and
`joblib>=1.4,<2`; the development acceptance run used scikit-learn 1.8.0.

## Historical replay checkpoint

`OpponentModel` schema version 4 retains a compact
`HistoricalHandCheckpoint`—hand key, historical model version, and archetype
prior—for every begun or committed hand. Replaying an old completed hand starts
from the posterior that existed when that hand began, even after newer evidence
has been committed. Concrete combo ranges are still limited to one active hand.

## Public action models

`ContextActionModel` and `HistoryAwareActionModel` are multinomial sklearn
logistic-regression pipelines. The `ColumnTransformer` uses
`OneHotEncoder(handle_unknown="ignore")` for categorical context and
`StandardScaler` for numeric inputs. Predictions are aligned to fold/check/call/
bet/raise, masked against the legal-action flags, and renormalized. If all raw
legal mass is zero, the documented fallback is uniform over legal actions.

Context inputs are exactly `OpponentFeatureVector`. Targets, chosen sizing,
public subject IDs, dataset/session/hand/sequence/group keys, profile labels,
cards, fingerprints, traces, future boards, and deck seeds never enter the
matrix.

History-aware rows add Bayesian posterior means and `log1p(opportunities)` for
VPIP, PFR, open raise, 3-bet, fold/call/raise versus bet, bet when checked to,
and aggression. The builder emits features from statistics through action
`t-1`, then observes action `t`. Histories are keyed by
`(dataset_session_id, public_subject_id)`, preventing cross-session aliasing.

## Evaluation

Both grouped random and per-session temporal splits keep hands and duplicate
correlation groups atomic. Shared metrics are multiclass log loss (also reported
as mean NLL), accuracy, macro F1, multiclass Brier score, ECE, reliability bins,
per-class frequencies, and public-context slices. Metric differences use a
correlation-group bootstrap, never an individual-decision bootstrap.

The shared comparison scores the legal-frequency, context logistic,
history-aware logistic, and existing Bayesian/archetype baseline on identical
held-out decisions. Logistic coefficients are inspectable associations—not
causal poker effects.

For the development fit, example largest associations included positive prior
voluntary raises for CALL (`+0.491`), negative prior voluntary raises for RAISE
(`-0.720`), positive prior 3-bet posterior mean for RAISE in the history model
(`+0.348`), and positive `can_check` for CHECK (`+0.411`). These describe the
fitted, correlated synthetic sample and must not be interpreted as causal poker
effects or universally correct strategy relationships.

## Privileged hand-conditioned model

`ResearchHandConditionedDataset` is explicitly synthetic and privileged. It
joins public decisions with research labels only while constructing training
rows. Candidate features are objective high/low rank, pair, suitedness, gap,
canonical class, made category, draws, and board interaction.

At inference, `LearnedRangeBelief` receives no true cards. It builds features for
every blocker-legal candidate combo, makes one batched `predict_proba` call, and
updates:

```text
P(h | action) proportional to P(h) × P_learned(action | h, public, history)
```

Sequential actions repeatedly apply this update and renormalize. Poker Coach
offers **Learned Model v2** only after a trusted local hand-conditioned artifact
is loaded; Manual and Opponent Model v1 remain available.

## Persistence and CLI

Artifacts contain the fitted preprocessing/classifier pipeline plus versioned
metadata: model/feature schema, dataset fingerprint, rows/groups, sklearn
version, action classes, metrics, and seed.

```bash
python3 -m poker_ai train-opponent-model \
  --hands 5000 --sessions-per-profile 10 --seed 42 \
  --model-type history --output artifacts/opponent_action_v2.joblib

python3 -m poker_ai evaluate-opponent-model \
  --model artifacts/opponent_action_v2.joblib --hands 1000 --seed 84
```

Joblib uses pickle internally and can execute code. `load_trusted_local_artifact`
is deliberately named: never load an artifact from an untrusted source.

## Development acceptance result

The checked-in machine report is `docs/MILESTONE7_RESULTS.json`. The balanced
run used 360 physical hands across six presets and two sessions per preset,
yielding 217 target decisions and 38 grouped held-out decisions.

| Model | Log loss | Accuracy | Macro F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Legal frequency | 1.098 | 0.605 | 0.227 | 0.608 | 0.166 |
| Context logistic | 4.223 | 0.684 | 0.404 | 0.562 | 0.174 |
| History logistic | 4.520 | 0.605 | 0.233 | 0.676 | 0.251 |
| Bayesian/archetype | 0.638 | 0.737 | 0.461 | 0.397 | 0.090 |

The context-frequency, history-context, and history-Bayesian grouped bootstrap
intervals all contained zero. The learned hand-conditioned range was also worse:
true-combo NLL 7.394 versus 6.905, equity MAE 0.118 versus 0.099, and equity RMSE
0.137 versus 0.115. Parameterized synthetic OOD log loss was 1.153 for context
and 1.631 for history.

This is a negative/null decision gate. Context logistic did not establish an
improvement over legal frequency; history did not improve context; neither beat
the Bayesian baseline; learned range and downstream equity did not improve.
The 1,800-hand exact-Bayesian run was stopped after several minutes, documenting
a scaling limit in the transparent baseline rather than a learned-model win.

Do not promote Learned Model v2 as the default. The next experiment should first
increase grouped sample size and cache/batch the expensive Bayesian likelihood
path. If nonlinear residual structure remains after that, evaluate sklearn
`HistGradientBoostingClassifier` before any neural or sequence model.
