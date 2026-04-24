"""
Environment verification script.

Run this first to confirm PyTorch, CUDA, and all dependencies are working.

Usage:
    python scripts/check_env.py
"""

import sys


def check_python():
    v = sys.version_info
    print(f"Python: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("  WARNING: Python 3.10+ recommended")
    else:
        print("  OK")


def check_torch():
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Quick sanity test
            x = torch.randn(2, 3).cuda()
            y = x @ x.T
            print(f"  GPU compute test: PASSED")
        else:
            print("  WARNING: No GPU detected. Training will be slow on CPU.")
        return True
    except ImportError:
        print("\nPyTorch: NOT INSTALLED")
        print("  Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_torchvision():
    try:
        import torchvision
        print(f"\nTorchvision: {torchvision.__version__}")

        # Verify ConvNeXt is available
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        print("  ConvNeXt_Tiny: available")
        return True
    except ImportError as e:
        print(f"\nTorchvision: ERROR — {e}")
        return False


def check_dependencies():
    deps = {
        "albumentations": "albumentations",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "PIL": "Pillow",
        "fastapi": "fastapi",
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
    }

    print("\nDependencies:")
    all_ok = True
    for import_name, pip_name in deps.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  {pip_name}: {version}")
        except ImportError:
            print(f"  {pip_name}: NOT INSTALLED  →  pip install {pip_name}")
            all_ok = False
    return all_ok


def main():
    print("=" * 50)
    print("Environment Check — Safety Violation Classifier")
    print("=" * 50)

    check_python()
    torch_ok = check_torch()
    if torch_ok:
        check_torchvision()
    deps_ok = check_dependencies()

    print("\n" + "=" * 50)
    if torch_ok and deps_ok:
        print("All checks passed. Ready to train.")
    else:
        print("Some dependencies missing. Install them and re-run.")
        print("  pip install -r requirements.txt")
    print("=" * 50)


if __name__ == "__main__":
    main()
