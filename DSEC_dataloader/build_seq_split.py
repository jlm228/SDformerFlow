"""Derive the single-chunk split CSVs from the published two-chunk ones.

DSECDatasetLite picks its split file from `data.num_chunks`:

    num_chunks == 2  ->  <split>_split_doubleseq.csv   (two columns: frame1, frame2)
    num_chunks == 1  ->  <split>_split_seq.csv         (one column)

Only the `doubleseq` variants exist -- they are vendored in this repo under `splits/` (see
splits/README.md for why they are not upstream). The SNN config uses `num_chunks: 1`, so the
single-chunk lists have to be derived.

Column 2 of the doubleseq CSV is the *target* frame -- the one whose ground truth is loaded
(see DSEC_dataset_lite.__getitem__, which reads the label from target_file_2 when
num_chunks == 2, and from target_file_1 when num_chunks == 1). Taking that column therefore
preserves the authors' exact train/valid partition rather than inventing a new one.

    python build_seq_split.py                      # both splits, default root
    python build_seq_split.py --root /path/to/saved_flow_data
"""

import argparse
import csv
import os


def build_seq_split(sequence_lists_dir: str, split: str, overwrite: bool = False) -> str:
    src = os.path.join(sequence_lists_dir, '{}_split_doubleseq.csv'.format(split))
    dst = os.path.join(sequence_lists_dir, '{}_split_seq.csv'.format(split))

    if not os.path.isfile(src):
        raise FileNotFoundError(
            '{} not found. Install the vendored splits first:\n'
            '    mkdir -p <saved_flow_data>/sequence_lists\n'
            '    cp splits/*.csv <saved_flow_data>/sequence_lists/'.format(src))

    if os.path.isfile(dst) and not overwrite:
        raise FileExistsError('{} already exists; pass --overwrite to replace it.'.format(dst))

    with open(src, newline='') as f:
        rows = [row for row in csv.reader(f) if row]

    targets = []
    for i, row in enumerate(rows, start=1):
        if len(row) < 2:
            raise ValueError(
                '{}:{} has {} column(s), expected 2. Is this really a doubleseq split?'.format(
                    src, i, len(row)))
        targets.append(row[1].strip())

    # Consecutive pairs overlap (…_0001,…_0002 then …_0002,…_0003), so column 2 is already
    # unique. Assert rather than silently dedupe -- a duplicate means the input is malformed.
    if len(set(targets)) != len(targets):
        raise ValueError('{} has duplicate entries in column 2.'.format(src))

    with open(dst, 'w', newline='') as f:
        csv.writer(f).writerows([t] for t in targets)

    print('{:>5} rows  {} -> {}'.format(len(targets), os.path.basename(src),
                                        os.path.basename(dst)))
    return dst


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='../data/Datasets/DSEC/saved_flow_data',
                   help='saved_flow_data directory (default: %(default)s)')
    p.add_argument('--splits', nargs='+', default=['train', 'valid'])
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    sequence_lists_dir = os.path.join(args.root, 'sequence_lists')
    for split in args.splits:
        build_seq_split(sequence_lists_dir, split, overwrite=args.overwrite)
