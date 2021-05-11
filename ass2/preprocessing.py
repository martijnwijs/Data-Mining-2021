# This file contains the main function for preprocessing 

from imputations import *
from feature_engineering import * 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    # Required parameters 

    parser.add_argument('--p', type=int, help='number of product', default=3)
    parser.add_argument('--C', type=list, help='cost per product', default=[1,1,1])
    parser.add_argument('--b', type=int, help='budget', default=2)
    parser.add_argument('--n', type=int, help='number of voters', default=3)
    parser.add_argument('--k', type=int, help='max number of products on a ballot', default=2)
    parser.add_argument('--sample_size', type=int, help='number of checks on strategyproofness', default=10)

    args = parser.parse_args()

    products = list(map(str, list(range(args.p))))
    costs = list(map(int, args.C))

    cost_dict = dict(zip(products, costs))
    possible_ballots_ = list(k_set(products, args.k))
    profile_ = random.choices(possible_ballots_, k=args.n)


    result = [check_strategyproofness(profile_, cost_dict, possible_ballots_, args.n, args.b) for i in range(args.sample_size)]
    print(f'The provided situation is strategyproof: {sum(result)/len(result) * 100}%')

