import subprocess
import sys

def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {command}")
        sys.exit(1)

def main():
    print("Starting environment setup...")

    # Recommended Python version check
    if sys.version_info < (3, 8) or sys.version_info >= (3, 12):
        print("⚠️ Warning: This setup script is tested and recommended for Python 3.8 to 3.11.")
        print(f"You are running Python {sys.version_info.major}.{sys.version_info.minor}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            print("Exiting setup. Please install a compatible Python version (3.8 - 3.11).")
            sys.exit(0)

    # First, uninstall potentially conflicting packages to avoid version conflicts
    print("Uninstalling existing packages if any...")
    uninstall_packages = [
        "tensorflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "flask",
        "pillow"
    ]
    for pkg in uninstall_packages:
        run_command(f"{sys.executable} -m pip uninstall -y {pkg}")

    # Now, reinstall all packages pinned to compatible versions
    print("Installing compatible package versions...")

    install_commands = [
        # Use TensorFlow with a stable compatible version with Python 3.8-3.11
        f"{sys.executable} -m pip install numpy==1.24.4",
        f"{sys.executable} -m pip install pandas==1.5.3",
        f"{sys.executable} -m pip install scikit-learn==1.2.2",
        f"{sys.executable} -m pip install flask==2.2.4",
        f"{sys.executable} -m pip install pillow==9.5.0",
        f"{sys.executable} -m pip install tensorflow==2.13.0"
    ]

    for cmd in install_commands:
        run_command(cmd)

    print("\n✅ All packages installed successfully with compatible versions.")
    print("You can now run your project without version conflicts.")
    print(f"Your Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("Finished environment setup.")

if __name__ == "__main__":
    main()

