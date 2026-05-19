"""
Run both experiments from Miller et al., IEEE MLSP 2017.

Usage:
    python run_all.py            # both experiments
    python run_all.py synthetic  # synthetic only (fast, ~5 min)
    python run_all.py mnist      # MNIST only (~60 min)
"""

import sys


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if target in ('all', 'synthetic'):
        from synthetic_exp import main as run_syn
        run_syn()

    if target in ('all', 'mnist'):
        from mnist_exp import main as run_mnist
        run_mnist()

    print('\nDone. Figures saved in figures/')


if __name__ == '__main__':
    main()
