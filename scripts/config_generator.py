import itertools
import json
import os


def generate_bambu_configs(enable_pipeline=False):
    configs = []
    config_id = 0

    clock_periods = [5, 8, 10, 15, 20]
    memory_policies = ['ALL_BRAM', 'NO_BRAM']
    channels_types = ['MEM_ACC_11', 'MEM_ACC_N1', 'MEM_ACC_NN']
    channels_numbers = [1, 2]

    # No-pipeline configs
    for cp, mem, ch_type, ch_num in itertools.product(
        clock_periods, memory_policies, channels_types, channels_numbers
    ):
        configs.append({'id': config_id, 'tool': 'bambu', 'clock_period': cp,
            'pipeline': False, 'pipeline_ii': None, 'memory_policy': mem,
            'channels_type': ch_type, 'channels_number': ch_num})
        config_id += 1

    # Pipeline configs (only if enabled)
    if enable_pipeline:
        pipeline_iis = [1, 2, 4]
        for cp, ii, mem, ch_type, ch_num in itertools.product(
            clock_periods, pipeline_iis, memory_policies,
            channels_types, channels_numbers
        ):
            configs.append({'id': config_id, 'tool': 'bambu', 'clock_period': cp,
                'pipeline': True, 'pipeline_ii': ii, 'memory_policy': mem,
                'channels_type': ch_type, 'channels_number': ch_num})
            config_id += 1

    return configs


def config_to_bambu_cmd(config, src_file, top_func):
    cmd = f"bambu {src_file} --top-fname={top_func}"
    cmd += f" --clock-period={config['clock_period']}"
    cmd += f" --memory-allocation-policy={config['memory_policy']}"
    cmd += f" --channels-type={config['channels_type']}"
    cmd += f" --channels-number={config['channels_number']}"
    if config['pipeline']:
        cmd += f" --pipelining={top_func}={config['pipeline_ii']}"
    cmd += " --no-clean"
    return cmd


def save_configs(configs, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"Saved {len(configs)} configurations to {filepath}")


if __name__ == "__main__":
    no_pipe = generate_bambu_configs(enable_pipeline=False)
    with_pipe = generate_bambu_configs(enable_pipeline=True)
    print(f"No pipeline: {len(no_pipe)} configs")
    print(f"With pipeline: {len(with_pipe)} configs")
