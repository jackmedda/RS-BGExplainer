import os
import pprint
import argparse

import pandas as pd


DEFAULT_AGE_RANGES = [
    "Under 18",
    "18-24",
    "25-34",
    "35-44",
    "45-49",
    "50-55",
    "56+"
]


DEFAULT_MAP_AGE_RANGES = [
    lambda x: x < 18,
    lambda x: 18 <= x <= 24,
    lambda x: 25 <= x <= 34,
    lambda x: 35 <= x <= 44,
    lambda x: 45 <= x <= 49,
    lambda x: 50 <= x <= 55,
    lambda x: x >= 56
]


def parse_age_range(str_ar: str):
    if '_' in str_ar:
        n1, n2 = map(int, str_ar.split('_'))
        return str_ar.replace('_', '-'), lambda x: n1 <= x <= n2
    else:
        func = "e" if '=' in str_ar else 't'
        pat = '=' if '=' in str_ar else ''
        func = "l" + func if '<' in str_ar else 'g' + func
        pat += '<' if '<' in str_ar else '>'

        number = str_ar.replace(pat, '')
        return f"Under {number}", lambda x: getattr(x, f"__{func}__")(int(number))


if __name__ == "__main__":
    r"""
    python -m src.data.map_multi2binary --uf sample.user --ouf dataset.user --ar <18 18_24 25_34 35_44 45_49 50_55 56>=
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_filepath', '--uf', required=True)
    parser.add_argument('--out_user_filepath', '--ouf', required=True)
    parser.add_argument('--age_ranges', '--ar', nargs='+', default=None)
    parser.add_argument('--age_col', default=None)

    args = parser.parse_args()

    if args.age_ranges is None:
        age_ranges = DEFAULT_AGE_RANGES
        map_age_ranges = DEFAULT_MAP_AGE_RANGES
    else:
        age_ranges, map_age_ranges = zip(*map(parse_age_range, args.age_ranges))

    user_df = pd.read_csv(args.user_filepath, sep='\t')

    pprint.pprint(vars(args))

    os.makedirs(args.out_folderpath, exist_ok=True)

    args.age_col = [col for col in user_df.columns if 'age' in col.lower()][0] if args.age_col is None else args.age_col
    map_age = {}
    for ar, mar in zip(age_ranges, map_age_ranges):
        mask = user_df[args.age_col].map(mar)
        ar_ages = user_df.loc[mask, args.age_col].unique()
        map_age.update(dict(zip(ar_ages, [ar] * len(ar_ages))))

    user_df[args.age_col] = user_df[args.age_col].map(map_age)
    user_df.to_csv(args.out_user_filepath, index=None, sep='\t')
