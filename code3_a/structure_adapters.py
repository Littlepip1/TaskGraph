"""Official communication STRUCTURE adapters, not full MARL reproductions.

HYGMA: invokes the unchanged official DynamicSpectralClustering class.
ExpoComm: invokes its unchanged get_exp_neighbors function with a minimal
NumPy tensor facade; no learned message encoder/GRU is claimed.
Legacy DHG: unchanged user-supplied selector, single-task only.
"""
from __future__ import annotations
import ast
from collections import deque
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings
import numpy as np
import tgexp
from tgexp.selection import select

ROOT = Path(__file__).resolve().parent


class TensorArray(np.ndarray):
    def expand(self, *sizes):
        return np.broadcast_to(self, tuple(old if new == -1 else new for old, new in zip(self.shape, sizes))).view(TensorArray)
    def long(self):
        return self.astype(np.int64)


class NumpyTorchFacade:
    @staticmethod
    def arange(*args): return np.arange(*args).view(TensorArray)
    @staticmethod
    def pow(a, b): return np.power(a, b).view(TensorArray)
    @staticmethod
    def zeros(*args): return np.zeros(*args).view(TensorArray)
    @staticmethod
    def cat(xs): return np.concatenate(xs).view(TensorArray)


def load_exponential_function():
    path = ROOT / 'official_modules/ExpoComm/ExpoComm_controller.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'get_exp_neighbors')
    namespace = {'th': NumpyTorchFacade}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['get_exp_neighbors']


class NumpyHistory:
    # Adapter to the official class's .view(...).cpu().numpy() interface.
    def __init__(self, array): self.array = np.asarray(array)
    @property
    def shape(self): return self.array.shape
    def view(self, *shape): return NumpyHistory(self.array.reshape(*shape))
    def cpu(self): return self
    def numpy(self): return self.array


def load_clustering_class():
    p = ROOT / 'official_modules/HYGMA/dynamic_clustering.py'
    spec = importlib.util.spec_from_file_location('_official_hygma_clustering', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DynamicSpectralClustering


class StructurePolicy:
    def __init__(self, job):
        self.job, self.kind = job, job['baseline_name']
        self.n = job['n']
        self.calls = self.excluded = self.updates = 0
        self.update_seconds = 0.
        self.last_tick = -1
        self.last_update = -job['period']
        self.groups = [list(range(self.n))]
        self.group_map = np.zeros(self.n, int)
        self.history = deque(maxlen=job.get('history_length', 8))
        self.warning_count = 0
        self.mask_sizes = []
        self.exp = load_exponential_function() if self.kind.startswith('expocomm') else None
        if self.kind == 'hygma_group':
            cls = load_clustering_class()
            self.clusterer = cls(2, min(job.get('max_clusters', max(4, 2*job['tasks'])), self.n - 1), self.n)
        if self.kind.startswith('legacy_dhg'):
            if job['tasks'] != 1:
                raise ValueError('Unmodified legacy selector supports single-task only')
            legacy = str(ROOT / 'legacy_dynamic')
            if legacy not in sys.path: sys.path.insert(0, legacy)
            from dynamic_hyperedge.selector import select_dynamic_hyperedge, DynamicHyperedgeConfig
            self.legacy_fn = select_dynamic_hyperedge
            self.legacy_config = DynamicHyperedgeConfig()

    def update(self, context):
        tick = context['tick']
        if tick == self.last_tick: return
        self.last_tick = tick
        if self.kind != 'hygma_group': return
        # The same benchmark estimates and declared role plan available to ours.
        # No target truth or future physical trajectory is included.
        features = np.c_[context['positions']/70., context['references']/70.,
                         context['current_roles'], context['future_roles']]
        self.history.append(features)
        if tick - self.last_update < self.job['period']: return
        array = np.stack(self.history, axis=1)  # agent x observed-time x features
        # Official spectral/kmeans RNG is global; seed it locally and restore it.
        state = np.random.get_state()
        clock = time.perf_counter()
        try:
            np.random.seed(np.random.SeedSequence([self.job['seed'], tick, 812]).generate_state(1)[0])
            with warnings.catch_warnings(record=True) as caught:
                updated, groups, moved = self.clusterer.update_groups(NumpyHistory(array), .6)
            self.warning_count += len(caught)
            if updated:
                self.groups = groups
                self.updates += 1
                for k, members in enumerate(groups): self.group_map[members] = k
        finally:
            np.random.set_state(state)
            self.update_seconds += time.perf_counter() - clock
        self.last_update = tick

    def __call__(self, c):
        self.calls += 1
        self.update(c)
        source, candidates = c['source'], c['candidates']
        if self.kind.startswith('legacy_dhg'):
            beliefs = [None] * self.n
            beliefs[source] = c['source_belief']
            for r, b in c['beliefs'].items(): beliefs[r] = b
            mode = 'additive' if self.kind == 'legacy_dhg_additive' else 'set_conditioned'
            result = self.legacy_fn(source, candidates, beliefs, c['positions'], c['offsets'],
                                    SimpleNamespace(**c['observation']), c['network'], c['payload_bits'],
                                    c['gate_decision']['reason'], c['gate_decision'], c['gate_cfg'],
                                    c['controller'], c['deadline'], mode=mode, cfg=self.legacy_config)
            return result['selected_recipients']
        if self.kind == 'hygma_group':
            allowed = {r for r in candidates if self.group_map[r] == self.group_map[source]}
        elif self.kind.startswith('expocomm'):
            degree = math.ceil(math.log2(self.n)) + 1
            neighbors = np.asarray(self.exp(1, self.n, degree))[0]
            if self.kind == 'expocomm_one_peer':
                # One model communication round per opportunity period.
                # Receiver rows list their incoming senders; invert for a source.
                column = (c['tick'] // c['period']) % degree
                allowed = set(np.flatnonzero(neighbors[:, column] == source).tolist())
            else:
                allowed = set(np.flatnonzero(np.any(neighbors == source, axis=1)).tolist())
            allowed.discard(source)
            allowed.intersection_update(candidates)
        else:
            raise ValueError(self.kind)
        self.excluded += len(candidates) - len(allowed)
        self.mask_sizes.append(len(allowed))
        # Identical downstream controller-impact rule and omission quota.
        values = {r: c['values'][r] for r in allowed}
        routes = {r: c['routes'][r] for r in allowed}
        return select(values, routes, c['payload_bits'], c['fraction'], c['absolute'])['selected']

    def observe_step(self, context):
        self.update(context)

    def summary(self):
        return dict(baseline=self.kind, policy_calls=self.calls, excluded_by_structure=self.excluded,
                    grouping_updates=self.updates, grouping_seconds=self.update_seconds,
                    final_group_sizes=[len(g) for g in self.groups], clustering_warnings=self.warning_count,
                    mean_allowed=float(np.mean(self.mask_sizes)) if self.mask_sizes else 0.,
                    scope='official structure module / legacy selector adaptation, not full trained MARL',
                    latent_message_model_reproduced=False)


def create_policy(job):
    return StructurePolicy(job)

