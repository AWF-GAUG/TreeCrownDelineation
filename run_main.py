import yaml
import itertools
import subprocess
import gc

def run_training(config_file, arch, backbone, max_epochs):
    # Load the YAML configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Update the configuration with the current combination
    config['training']['arch'] = arch
    config['training']['backbone'] = backbone
    config['training']['max_epochs'] = max_epochs

    # Save the temporary configuration
    temp_config_path = './temp_config.yaml'
    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)

    # Run the training script with the temporary configuration
    print(f"\n\nSTARTING WITH TRAINING: {arch} | {backbone} | {max_epochs}\n\n")
    subprocess.run(['python', 'multicls_raster_script.py', '--config', temp_config_path])
    gc.collect()
    print(f"❗training complete for model: {arch} with backbone: {backbone} for {max_epochs} epochs ✅")

if __name__ == "__main__":
    config_file = './multicls_config.yaml'

    # Load the configuration to get options
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Iterate over all combinations and run training
    for idx, (arch, backbone, max_epochs) in enumerate(itertools.product(
            config['training']['arch_options'], 
            config['training']['backbone_options'], 
            config['training']['max_epochs_options'])):
        # print(idx, arch, backbone, max_epochs)
        run_training(config_file, arch, backbone, max_epochs)
