import random


def grid_search(configs):
    return configs


def random_search(configs, sample_ratio=0.3, seed=42):
    random.seed(seed)
    sample_size = max(1, int(len(configs) * sample_ratio))
    return random.sample(configs, min(sample_size, len(configs)))


def latin_hypercube_search(configs, param_keys, sample_ratio=0.2, seed=42):
    random.seed(seed)
    sample_size = max(1, int(len(configs) * sample_ratio))
    param_values = {}
    for key in param_keys:
        values = list(set(c[key] for c in configs if c.get(key) is not None))
        values.sort(key=str)
        param_values[key] = values
    selected = []
    selected_ids = set()
    for key, values in param_values.items():
        for val in values:
            candidates = [c for c in configs
                         if c.get(key) == val and c['id'] not in selected_ids]
            if candidates:
                chosen = random.choice(candidates)
                selected.append(chosen)
                selected_ids.add(chosen['id'])
    remaining = [c for c in configs if c['id'] not in selected_ids]
    extra_needed = max(0, sample_size - len(selected))
    if extra_needed > 0 and remaining:
        extra = random.sample(remaining, min(extra_needed, len(remaining)))
        selected.extend(extra)
    return selected


if __name__ == "__main__":
    from config_generator import generate_bambu_configs
    configs, _ = generate_bambu_configs()
    print(f"Total configs: {len(configs)}")
    print(f"Grid search: {len(grid_search(configs))} configs")
    print(f"Random 30%: {len(random_search(configs, 0.3))} configs")
    param_keys = ['clock_period', 'pipeline', 'memory_policy', 'channels_type']
    print(f"LHS 20%: {len(latin_hypercube_search(configs, param_keys, 0.2))} configs")
