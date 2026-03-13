import argparse
import os


def _run_baseline(args):
    import baseline.train_baseline_cv as b

    b.RUNS_ROOT = args.root
    b.EXP_NAME = 'baseline'
    if args.fast:
        b.NUM_EPOCHS = 1
        b.NUM_FOLDS = 2
        b.PATIENCE = 1
        b.NUM_WORKERS = 0
        b.BATCH_SIZE = 64
        b.ACCUMULATION_STEPS = 1
        b.IMAGE_SIZE = 96
    b.main()


def _run_dfs(args, exp_name, focus_mode, att_loss_weight):
    import train_dynamic_focus as d

    d.RUNS_ROOT = args.root
    d.EXP_NAME = exp_name
    d.FOCUS_MODE = focus_mode
    d.ATT_LOSS_WEIGHT = float(att_loss_weight)
    if args.fast:
        d.NUM_EPOCHS = 1
        d.NUM_FOLDS = 2
        d.PATIENCE = 1
        d.NUM_WORKERS = 0
        d.BATCH_SIZE = 8
        d.ACCUMULATION_STEPS = 1
    d.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='ablation_outputs')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument(
        '--experiments',
        nargs='*',
        default=['baseline', 'dfs_center', 'dfs_no_att', 'dfs_full'],
    )
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)

    for name in args.experiments:
        if name == 'baseline':
            _run_baseline(args)
        elif name == 'dfs_center':
            _run_dfs(args, exp_name='dfs_center', focus_mode='center', att_loss_weight=0.0)
        elif name == 'dfs_no_att':
            _run_dfs(args, exp_name='dfs_no_att', focus_mode='attn', att_loss_weight=0.0)
        elif name == 'dfs_full':
            _run_dfs(args, exp_name='dfs_full', focus_mode='attn', att_loss_weight=0.2)
        else:
            raise ValueError(f'Unknown experiment: {name}')


if __name__ == '__main__':
    main()

