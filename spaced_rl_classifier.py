from spaced_rl_classifier import SpacedRLClassifier, SimpleMLP, RLSchedulerPolicy

# buat model & policy
model = SimpleMLP(dim=n_features, hidden=128, n_classes=n_classes)
policy = RLSchedulerPolicy(n_classes=n_classes)

# inisialisasi classifier
clf = SpacedRLClassifier(model=model, policy=policy, device=device)

# fit
clf.fit(items, train_count=train_count, epochs=80, batch_size=64, k_retrieval=3)

# evaluasi
eval_res = clf.evaluate(list(range(train_count, len(items))))
print(eval_res["report"])
