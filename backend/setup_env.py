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

    # Upgrade pip to latest version first for best compatibility
    print("Upgrading pip to the latest version...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Uninstall existing packages if any
    print("Uninstalling existing packages if any...")
    uninstall_packages = [
        "tensorflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "flask",
        "pillow",
        "flask-cors"
    ]
    for pkg in uninstall_packages:
        run_command(f"{sys.executable} -m pip uninstall -y {pkg}")

    # Install compatible package versions including flask-cors
    print("Installing compatible package versions...")

    install_commands = [
        f"{sys.executable} -m pip install numpy==1.24.4",
        f"{sys.executable} -m pip install pandas==1.5.3",
        f"{sys.executable} -m pip install scikit-learn==1.2.2",
        f"{sys.executable} -m pip install flask==2.2.4",
        f"{sys.executable} -m pip install pillow==9.5.0",
        f"{sys.executable} -m pip install flask-cors==3.1.0",
        f"{sys.executable} -m pip install tensorflow==2.13.0"
    ]

    for cmd in install_commands:
        run_command(cmd)

    print("\n✅ All packages installed successfully with compatible versions.")
    print(f"Your Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("Finished environment setup.")

if __name__ == "__main__":
    main()
