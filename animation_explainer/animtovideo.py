import subprocess
import sys
import yaml

def main():
    print("Starting Manim animation render...")

    # Read custom config
    with open("custom_config.yml", "r") as f:
        config = yaml.safe_load(f)

    cli_cfg = config.get("CLI", {})

    # Base command
    command = [
        sys.executable, "-m", "manim",
        "animation.py",
        "MambaJEPAExplainer"
    ]

    # Apply configurations from yaml
    quality_map = {
        "low_quality": "-ql",
        "medium_quality": "-qm",
        "high_quality": "-qh",
        "production_quality": "-qp",
        "fourk_quality": "-qk"
    }
    quality = cli_cfg.get("quality", "high_quality")
    if quality in quality_map:
        command.append(quality_map[quality])

    if "frame_rate" in cli_cfg:
        command.extend(["--fps", str(cli_cfg["frame_rate"])])

    if "pixel_width" in cli_cfg and "pixel_height" in cli_cfg:
        command.extend(["--resolution", f"{cli_cfg['pixel_width']},{cli_cfg['pixel_height']}"])

    if "background_color" in cli_cfg:
        command.extend(["-c", cli_cfg["background_color"]])

    if "verbosity" in cli_cfg:
        command.extend(["-v", cli_cfg["verbosity"]])

    print(f"Running command: {' '.join(command)}")

    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Manim render completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during Manim rendering:")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
