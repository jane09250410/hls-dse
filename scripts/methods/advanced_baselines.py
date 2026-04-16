#!/usr/bin/env python3
"""
advanced_baselines.py — Strong DSE baselines for comparison with PA-DSE.

All four baselines fit the DSEMethod interface and share the same
config space, budget, seed, and synthesize_fn as PA-DSE.

Methods:
  SimulatedAnnealingMethod   — geometric cooling, neighbor-by-1-param
  GeneticAlgorithmMethod     — NSGA-II lite, tournament + uniform crossover
  GPBayesOptMethod           — GP surrogate + Expected Improvement, penalty for failures
  RFClassifierMethod         — random forest feasibility predictor (AutoScaleDSE-style)

Design: each method uses apply_reorder() to put its best candidate at front
of the queue, then select_next() pops it. This is the simplest way to fit
the streaming iteration protocol.
"""

import re
import random
import numpy as np
from collections import defaultdict
from typing import List, Optional, Dict, Any

from methods.base import DSEMethod, Config


# ──────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────

def extract_qor(output: str) -> Optional[float]:
    """Extract area (Bambu) or component count (Dynamatic)."""
    m = re.search(r"Total\s+estimated\s+area\s*[=:]\s*([\d.]+)", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"components\s*=\s*(\d+)", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


class ConfigEncoder:
    """One-hot encode categorical config parameters for ML models."""

    def __init__(self, configs: List[Config]):
        exclude = {"id", "tool"}
        all_keys = set()
        for c in configs:
            for k in c:
                if k not in exclude:
                    all_keys.add(k)
        self.keys = sorted(all_keys)

        self.value_map: Dict[str, List[str]] = {}
        for k in self.keys:
            values = set()
            for c in configs:
                v = c.get(k)
                if v is not None:
                    values.add(str(v))
            self.value_map[k] = sorted(values)

        self.dim = sum(len(v) for v in self.value_map.values())

    def encode(self, config: Config) -> np.ndarray:
        vec = []
        for k in self.keys:
            v = str(config.get(k, ""))
            for val in self.value_map[k]:
                vec.append(1.0 if v == val else 0.0)
        return np.array(vec, dtype=np.float64)

    def encode_all(self, configs: List[Config]) -> np.ndarray:
        return np.stack([self.encode(c) for c in configs])


# ──────────────────────────────────────────────────────────
# 1. Simulated Annealing
# ──────────────────────────────────────────────────────────

class SimulatedAnnealingMethod(DSEMethod):
    """Simulated Annealing on discrete config space.

    Starts from a random config. Each step:
      1. Propose neighbor (differ in 1 parameter value)
      2. Accept if better, else with prob exp(-delta/T)
      3. Cool temperature geometrically

    Failed synthesis gets penalty score (worst observed + 1).
    """

    def __init__(self, configs, benchmark_name, tool, budget, seed=None,
                 T_init=1000.0, T_final=1.0, **kwargs):
        super().__init__(configs, benchmark_name, tool, budget, seed=seed)
        self.T_init = T_init
        self.T_final = T_final
        self.rng = random.Random(seed if seed is not None else 0)
        self.configs_by_id = {c["id"]: c for c in configs}
        self.evaluated = set()
        self.current_config: Optional[Config] = None
        self.current_score: float = float("inf")
        self.worst_feasible_score = 0.0
        self.step = 0

    @property
    def method_name(self):
        return "SimulatedAnnealing"

    def _temperature(self) -> float:
        if self.budget <= 1:
            return self.T_final
        frac = self.step / max(self.budget - 1, 1)
        return self.T_init * ((self.T_final / self.T_init) ** frac)

    def _score(self, success: bool, qor: Optional[float]) -> float:
        if success and qor is not None:
            self.worst_feasible_score = max(self.worst_feasible_score, qor)
            return qor
        return self.worst_feasible_score + 1.0 if self.worst_feasible_score > 0 else 1e6

    def _find_neighbor(self, config: Config) -> Optional[Config]:
        exclude = {"id", "tool"}
        keys = [k for k in config if k not in exclude]
        candidates = []
        for c in self.configs:
            if c["id"] == config["id"] or c["id"] in self.evaluated:
                continue
            diff = sum(1 for k in keys if str(config.get(k)) != str(c.get(k)))
            if diff == 1:
                candidates.append(c)
        if candidates:
            return self.rng.choice(candidates)
        # no neighbors left → pick any unevaluated
        remaining = [c for c in self.configs if c["id"] not in self.evaluated]
        return self.rng.choice(remaining) if remaining else None

    def initialize(self) -> List[Config]:
        q = list(self.configs)
        self.rng.shuffle(q)
        return q

    def apply_reorder(self, queue: List[Config]) -> List[Config]:
        if self.current_config is None:
            return queue
        nxt = self._find_neighbor(self.current_config)
        if nxt is None:
            return queue
        rest = [c for c in queue if c["id"] != nxt["id"]]
        return [nxt] + rest

    def update(self, config, success, output, synthesis_time):
        self.step += 1
        self.evaluated.add(config["id"])
        qor = extract_qor(output)
        new_score = self._score(success, qor)

        if self.current_config is None:
            self.current_config = config
            self.current_score = new_score
            return

        delta = new_score - self.current_score
        T = self._temperature()
        if delta < 0:
            accept = True
        else:
            accept = self.rng.random() < np.exp(-delta / max(T, 1e-9))
        if accept:
            self.current_config = config
            self.current_score = new_score


# ──────────────────────────────────────────────────────────
# 2. Genetic Algorithm (NSGA-II lite, single-objective on area)
# ──────────────────────────────────────────────────────────

class GeneticAlgorithmMethod(DSEMethod):
    """Genetic Algorithm with tournament selection and uniform crossover.

    Maintains a population. Each generation:
      1. Evaluate all individuals (happens via streaming)
      2. Tournament select parents
      3. Uniform crossover + mutation → offspring
      4. Next generation = best half + offspring

    Failed synthesis gets worst-feasible + 1 penalty.
    """

    def __init__(self, configs, benchmark_name, tool, budget, seed=None,
                 pop_size=10, mutation_rate=0.1, tournament_k=3, **kwargs):
        super().__init__(configs, benchmark_name, tool, budget, seed=seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.rng = random.Random(seed if seed is not None else 0)
        self.configs_by_id = {c["id"]: c for c in configs}
        self.evaluated = set()
        self.scores: Dict[int, float] = {}
        self.worst_feasible_score = 0.0
        # Population management
        self.population: List[Config] = []
        self.offspring_queue: List[Config] = []
        # Build value pool for mutation
        self._build_value_pool(configs)

    @property
    def method_name(self):
        return "GeneticAlgorithm"

    def _build_value_pool(self, configs):
        exclude = {"id", "tool"}
        self.value_pool: Dict[str, List[Any]] = defaultdict(set)
        for c in configs:
            for k, v in c.items():
                if k not in exclude:
                    self.value_pool[k].add(v)
        self.value_pool = {k: list(v) for k, v in self.value_pool.items()}

    def _score(self, success, qor):
        if success and qor is not None:
            self.worst_feasible_score = max(self.worst_feasible_score, qor)
            return qor
        return self.worst_feasible_score + 1.0 if self.worst_feasible_score > 0 else 1e6

    def _find_matching_config(self, params: dict) -> Optional[Config]:
        """Find an unevaluated config matching the given parameters."""
        exclude = {"id", "tool"}
        for c in self.configs:
            if c["id"] in self.evaluated:
                continue
            if all(str(c.get(k)) == str(params.get(k)) for k in params if k not in exclude):
                return c
        return None

    def _crossover(self, p1: Config, p2: Config) -> dict:
        exclude = {"id", "tool"}
        child = {}
        for k in p1:
            if k in exclude:
                child[k] = p1[k]
            else:
                child[k] = p1[k] if self.rng.random() < 0.5 else p2[k]
        return child

    def _mutate(self, params: dict) -> dict:
        exclude = {"id", "tool"}
        mutated = dict(params)
        for k in mutated:
            if k in exclude:
                continue
            if self.rng.random() < self.mutation_rate:
                mutated[k] = self.rng.choice(self.value_pool[k])
        return mutated

    def _tournament(self) -> Config:
        k = min(self.tournament_k, len(self.population))
        cand = self.rng.sample(self.population, k)
        return min(cand, key=lambda c: self.scores.get(c["id"], float("inf")))

    def _breed_next_generation(self):
        """Generate offspring for next batch."""
        self.offspring_queue = []
        attempts = 0
        while len(self.offspring_queue) < self.pop_size and attempts < self.pop_size * 10:
            attempts += 1
            p1 = self._tournament()
            p2 = self._tournament()
            child = self._mutate(self._crossover(p1, p2))
            match = self._find_matching_config(
                {k: v for k, v in child.items() if k not in ("id", "tool")})
            if match is not None and match["id"] not in self.evaluated:
                if not any(c["id"] == match["id"] for c in self.offspring_queue):
                    self.offspring_queue.append(match)

        # Select survivors: top half of current population
        self.population.sort(key=lambda c: self.scores.get(c["id"], float("inf")))
        self.population = self.population[: self.pop_size // 2] + self.offspring_queue

    def initialize(self) -> List[Config]:
        # Initial population = random pop_size configs
        q = list(self.configs)
        self.rng.shuffle(q)
        self.population = q[: self.pop_size]
        self.offspring_queue = list(self.population)
        return q

    def apply_reorder(self, queue: List[Config]) -> List[Config]:
        # Breed next generation if offspring queue is empty
        if not self.offspring_queue and len(self.scores) >= self.pop_size:
            self._breed_next_generation()
        if not self.offspring_queue:
            return queue

        # Move first unevaluated offspring to front
        for candidate in self.offspring_queue:
            if candidate["id"] in self.evaluated:
                continue
            rest = [c for c in queue if c["id"] != candidate["id"]]
            return [candidate] + rest
        return queue

    def update(self, config, success, output, synthesis_time):
        self.evaluated.add(config["id"])
        qor = extract_qor(output)
        self.scores[config["id"]] = self._score(success, qor)
        # Remove from offspring queue
        self.offspring_queue = [c for c in self.offspring_queue
                                if c["id"] != config["id"]]


# ──────────────────────────────────────────────────────────
# 3. GP-based Bayesian Optimization
# ──────────────────────────────────────────────────────────

class GPBayesOptMethod(DSEMethod):
    """Gaussian Process Bayesian Optimization with Expected Improvement.

    Uses one-hot encoding for categorical parameters.
    Failed synthesis gets worst-feasible + 1 penalty (standard approach).
    Initial phase: n_init random samples, then EI-driven selection.

    Note: GP is computationally expensive; apply_reorder is O(|queue| × N_obs²).
    """

    def __init__(self, configs, benchmark_name, tool, budget, seed=None,
                 n_init=5, **kwargs):
        super().__init__(configs, benchmark_name, tool, budget, seed=seed)
        self.n_init = n_init
        self.rng = np.random.RandomState(seed if seed is not None else 0)
        self.encoder = ConfigEncoder(configs)
        self.observed: Dict[int, tuple] = {}  # id -> (success, qor, vector)
        self.worst_feasible = 0.0
        self.step = 0

    @property
    def method_name(self):
        return "GP-BO"

    def _score(self, success, qor):
        if success and qor is not None:
            self.worst_feasible = max(self.worst_feasible, qor)
            return qor
        return self.worst_feasible + 1.0 if self.worst_feasible > 0 else 1e6

    def initialize(self) -> List[Config]:
        q = list(self.configs)
        self.rng.shuffle(q)
        return list(q)

    def apply_reorder(self, queue):
        if self.step < self.n_init or len(self.observed) < 3:
            return queue

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
        except ImportError:
            return queue

        X, y = [], []
        for cid, (succ, qor, vec) in self.observed.items():
            X.append(vec)
            y.append(self._score(succ, qor))
        X = np.array(X)
        y = np.array(y)

        # Normalize y (GP works better)
        y_mean = y.mean()
        y_std = max(y.std(), 1e-6)
        y_norm = (y - y_mean) / y_std

        # GP: minimize score (so negate for maximization formulation of EI)
        kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3,
                                       n_restarts_optimizer=2,
                                       random_state=self.rng,
                                       normalize_y=False)
        try:
            gp.fit(X, -y_norm)  # maximize negative score
        except Exception:
            return queue

        queue_vecs = np.array([self.encoder.encode(c) for c in queue])
        mu, sigma = gp.predict(queue_vecs, return_std=True)

        # Expected improvement
        y_best = -y_norm.min()  # best observed (in negated space)
        sigma = np.maximum(sigma, 1e-9)
        z = (mu - y_best) / sigma
        ei = sigma * (z * _std_normal_cdf(z) + _std_normal_pdf(z))

        order = np.argsort(-ei)
        return [queue[i] for i in order]

    def update(self, config, success, output, synthesis_time):
        self.step += 1
        qor = extract_qor(output)
        vec = self.encoder.encode(config)
        self.observed[config["id"]] = (success, qor, vec)


def _std_normal_cdf(z):
    return 0.5 * (1 + _erf_approx(z / np.sqrt(2)))

def _std_normal_pdf(z):
    return np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)

def _erf_approx(x):
    # Abramowitz & Stegun approximation (accurate to ~1.5e-7)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


# ──────────────────────────────────────────────────────────
# 4. Random Forest Classifier (AutoScaleDSE-style)
# ──────────────────────────────────────────────────────────

class RFClassifierMethod(DSEMethod):
    """Random Forest feasibility classifier, similar to AutoScaleDSE.

    Phase 1: Evaluate n_init configs randomly to collect training data.
    Phase 2: Train RF(feasible vs infeasible), rank remaining queue
             by predicted P(feasible). Retrain every `retrain_every` steps.
    Among feasible candidates, prefer low predicted QoR (via regression).
    """

    def __init__(self, configs, benchmark_name, tool, budget, seed=None,
                 n_init=10, retrain_every=5, **kwargs):
        super().__init__(configs, benchmark_name, tool, budget, seed=seed)
        self.n_init = n_init
        self.retrain_every = retrain_every
        self.rng = np.random.RandomState(seed if seed is not None else 0)
        self.encoder = ConfigEncoder(configs)
        self.observed: Dict[int, tuple] = {}
        self.step = 0
        self.last_retrain = 0
        self.clf = None
        self.reg = None

    @property
    def method_name(self):
        return "RF_Classifier"

    def initialize(self) -> List[Config]:
        q = list(self.configs)
        self.rng.shuffle(q)
        return list(q)

    def _train(self):
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError:
            return

        X, y_cls, X_reg, y_reg = [], [], [], []
        for cid, (succ, qor, vec) in self.observed.items():
            X.append(vec)
            y_cls.append(1 if succ else 0)
            if succ and qor is not None:
                X_reg.append(vec)
                y_reg.append(qor)

        X = np.array(X)
        y_cls = np.array(y_cls)

        # Need both classes for RF classifier
        if len(set(y_cls)) >= 2:
            self.clf = RandomForestClassifier(
                n_estimators=50, max_depth=8,
                random_state=self.rng.randint(1 << 30))
            self.clf.fit(X, y_cls)
        else:
            self.clf = None

        if len(X_reg) >= 3:
            self.reg = RandomForestRegressor(
                n_estimators=50, max_depth=8,
                random_state=self.rng.randint(1 << 30))
            self.reg.fit(np.array(X_reg), np.array(y_reg))
        else:
            self.reg = None

    def apply_reorder(self, queue):
        if self.step < self.n_init:
            return queue

        if self.step - self.last_retrain >= self.retrain_every or self.clf is None:
            self._train()
            self.last_retrain = self.step

        if self.clf is None:
            return queue

        qv = np.array([self.encoder.encode(c) for c in queue])
        try:
            # P(class=1) = P(feasible)
            classes = list(self.clf.classes_)
            if 1 in classes:
                p_feasible = self.clf.predict_proba(qv)[:, classes.index(1)]
            else:
                p_feasible = np.zeros(len(qv))
        except Exception:
            return queue

        # Among predicted-feasible configs, prefer low QoR prediction
        if self.reg is not None:
            try:
                qor_pred = self.reg.predict(qv)
            except Exception:
                qor_pred = np.zeros(len(qv))
        else:
            qor_pred = np.zeros(len(qv))

        # Composite score: high P(feasible), low QoR. Sort descending by P - lambda*normalized_qor
        if qor_pred.max() > qor_pred.min():
            qor_norm = (qor_pred - qor_pred.min()) / (qor_pred.max() - qor_pred.min())
        else:
            qor_norm = np.zeros_like(qor_pred)
        score = p_feasible - 0.3 * qor_norm
        order = np.argsort(-score)
        return [queue[i] for i in order]

    def update(self, config, success, output, synthesis_time):
        self.step += 1
        qor = extract_qor(output)
        vec = self.encoder.encode(config)
        self.observed[config["id"]] = (success, qor, vec)
