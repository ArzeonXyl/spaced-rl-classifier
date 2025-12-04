# spaced_rl_classifier/__init__.py
import os, random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Item state & RL scheduler
# ----------------------------
@dataclass
class ItemState:
    idx: int
    topic: int
    reviews: int = 0
    last_seen_step: int = -1
    next_due: int = 0
    difficulty: float = 1.0
    ema_perf: float = 0.5

class RLSchedulerPolicy(nn.Module):
    def __init__(self, n_classes, emb_dim=8, hidden=64, n_actions=5):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
        self.n_actions = n_actions

    def forward(self, class_id, state_vec):
        e = self.emb(class_id)
        x = torch.cat([e, state_vec], dim=1)
        return self.net(x)

class RLScheduler:
    def __init__(self, states: List[ItemState], policy_net: RLSchedulerPolicy,
                 interval_actions=[1,3,5,10,20], device='cpu'):
        self.states = states
        self.policy = policy_net.to(device)
        self.interval_actions = interval_actions
        self.device = device
        self.saved_logprobs = []
        self.rewards = []

    def get_due(self, step:int):
        return [s for s in self.states if s.next_due <= step]

    def sample_batch(self, step:int, batch_size:int):
        due = self.get_due(step)
        if not due:
            remaining = sorted(self.states, key=lambda s: -s.difficulty)
            return remaining[:batch_size]
        by_topic = defaultdict(list)
        for s in due:
            by_topic[s.topic].append(s)
        topics = list(by_topic.keys())
        batch = []
        t_i = 0
        while len(batch) < batch_size and topics:
            t = topics[t_i % len(topics)]
            if by_topic[t]:
                batch.append(by_topic[t].pop(random.randrange(len(by_topic[t]))))
            else:
                topics.remove(t)
                t_i -= 1
            t_i += 1
        if len(batch) < batch_size:
            remaining = [s for s in self.states if s not in batch]
            remaining.sort(key=lambda x: -x.difficulty)
            batch += remaining[:batch_size - len(batch)]
        return batch

    def choose_interval_and_log(self, state: ItemState):
        time_since = 0
        if state.last_seen_step >= 0:
            time_since = (state.next_due - state.last_seen_step)
        vec = torch.tensor([state.difficulty, state.ema_perf, float(state.reviews), float(time_since)],
                           dtype=torch.float32).unsqueeze(0).to(self.device)
        class_id = torch.tensor([state.topic], dtype=torch.long).to(self.device)
        logits = self.policy(class_id, vec)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        self.saved_logprobs.append(logprob)
        return self.interval_actions[action.item()]

    def update_after_review(self, state: ItemState, step:int, success:bool):
        alpha = 0.2
        state.ema_perf = alpha * float(success) + (1-alpha) * state.ema_perf
        state.difficulty = max(0.01, 1.0 - state.ema_perf)
        state.reviews += 1 if success else max(0, state.reviews-1)
        state.last_seen_step = step
        state.next_due = step + self.choose_interval_and_log(state)
        r = 1.0 if success else -1.0
        self.rewards.append(r)

    def reinforce_update(self, optimizer, gamma=0.99):
        if not self.rewards or not self.saved_logprobs:
            return 0.0
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = 0
        for logprob, R in zip(self.saved_logprobs, returns):
            loss -= logprob * R
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.saved_logprobs = []
        self.rewards = []
        return float(loss.item())

# ----------------------------
# Simple MLP for classification
# ----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, dim, hidden=128, n_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# SpacedRLClassifier API
# ----------------------------
class SpacedRLClassifier:
    def __init__(self, model=None, policy=None, device='cpu'):
        self.device = device
        self.model = model
        self.policy = policy
        self.scaler = None
        self.items = None
        self.states = None
        self.train_count = None

    @staticmethod
    def preprocess_tabular(df: pd.DataFrame, target_col: str):
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].values
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
        if num_cols:
            X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        if cat_cols:
            X[cat_cols] = X[cat_cols].fillna("missing")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        scaler = None
        if num_cols:
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])
        return X, y, scaler

    @staticmethod
    def build_items(X, y) -> list:
        """Convert DataFrame / Series / numpy ke items [(tensor_feat, label, topic), ...]"""
        import torch
        import numpy as np
        import pandas as pd

        # pastikan X numeric
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X = X.fillna(0.0)
            X_array = X.values
        elif isinstance(X, pd.Series):
            X_array = X.to_frame().values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            X_array = X.astype(np.float32)
        else:
            X_array = X.float()  # kalau tensor

        # y
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_array = y.values.astype(int)
        elif isinstance(y, np.ndarray):
            y_array = y.astype(int)
        else:
            y_array = np.array(list(map(int, y)))

        X_tensor = torch.tensor(X_array, dtype=torch.float32)
        y_list = y_array.tolist()

        return [(X_tensor[i], int(y_list[i]), int(y_list[i])) for i in range(len(y_list))]

    def fit(self, items:List[Tuple[torch.Tensor,int,int]], train_count:int, epochs=50, batch_size=32,
            k_retrieval=3, lr=1e-3, lr_policy=1e-4):
        self.items = items
        self.train_count = train_count
        self.model.to(self.device)
        self.policy.to(self.device)
        self.states = [ItemState(idx=i, topic=items[i][2]) for i in range(len(items))]
        scheduler = RLScheduler(self.states[:train_count], self.policy, device=self.device)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        opt_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        criterion = nn.CrossEntropyLoss()
        global_step = 0

        for ep in range(epochs):
            steps_per_epoch = max(60, len(items) // batch_size)
            for _ in range(steps_per_epoch):
                batch_states = scheduler.sample_batch(global_step, batch_size)
                xs = torch.stack([items[s.idx][0] for s in batch_states]).to(self.device)
                ys = torch.tensor([items[s.idx][1] for s in batch_states], dtype=torch.long).to(self.device)

                # retrieval
                self.model.eval()
                success_flags = []
                with torch.no_grad():
                    for i, s in enumerate(batch_states):
                        item_success = False
                        for _ in range(k_retrieval):
                            logits = self.model(xs[i].unsqueeze(0))
                            probs = torch.softmax(logits, dim=1)
                            pred = torch.argmax(probs, dim=1).item()
                            if pred == ys[i].item() and float(probs.max().item()) > 0.6:
                                item_success = True
                                break
                        success_flags.append(item_success)

                # supervised update
                self.model.train()
                logits = self.model(xs)
                loss = criterion(logits, ys)
                opt.zero_grad()
                loss.backward()
                opt.step()

                # update scheduler
                for s_obj, succ in zip(batch_states, success_flags):
                    scheduler.update_after_review(s_obj, global_step, succ)

                if global_step % 50 == 0 and scheduler.rewards:
                    scheduler.reinforce_update(opt_policy)

                global_step += 1

        return self

    def predict(self, items_idx:List[int]):
        self.model.eval()
        X_tensor = torch.stack([self.items[i][0] for i in items_idx]).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def evaluate(self, items_idx:List[int]):
        preds = self.predict(items_idx)
        y_true = [self.items[i][1] for i in items_idx]
        report = classification_report(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)
        return {"accuracy": acc, "report": report, "preds": preds}
