# This file is part of visvalingamwyatt.
# https://github.com/fitnr/visvalingamwyatt

# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, 2017, fitnr <contact@fakeisthenewreal.org>

import argparse
import json

from .visvalingamwyatt import simplify_feature

try:
    import fiona

    inputhelp = 'geodata file'

    def simplify(inp, output, **kwargs):
        if output == '/dev/stdout':
            output = '/vsistdout'

        if inp == '/dev/stdin':
            output = '/vsistdin'

        with fiona.drivers():
            with fiona.open(inp, 'r') as src:
                with fiona.open(
                    output, 'w', schema=src.schema, driver=src.driver, crs=src.crs
                ) as sink:
                    for f in src:
                        sink.write(simplify_feature(f, **kwargs))

except ImportError:
    inputhelp = 'geojson file'

    def simplify(inp, output, **kwargs):
        with open(inp, 'r') as f:
            geojson = json.load(f)

            with open(output, 'w') as g:
                geojson['features'] = [
                    simplify_feature(f, **kwargs) for f in geojson['features']
                ]
                json.dump(geojson, g)


def main():
    parser = argparse.ArgumentParser(
        'simplify',
        description='Simplify geospatial data using with the Visvalingam-Wyatt algorithm',
    )
    parser.add_argument('input', default='/dev/stdin', help=inputhelp)

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-t', '--threshold', type=float, metavar='float', help='minimum area'
    )
    group.add_argument(
        '-n', '--number', type=int, metavar='int', help='number of points to keep'
    )
    group.add_argument(
        '-r',
        '--ratio',
        type=float,
        metavar='float',
        help='fraction of points to keep (default: 0.90)',
    )

    parser.add_argument('-o', '--output', metavar='file', default='/dev/stdout')

    args = parser.parse_args()

    if args.input == '-':
        args.input = '/dev/stdin'

    if args.output == '-':
        args.output = '/dev/stdout'

    kwargs = {}

    if args.number:
        kwargs['number'] = args.number

    elif args.threshold:
        kwargs['threshold'] = args.threshold

    else:
        kwargs['ratio'] = args.ratio or 0.90

    simplify(args.input, args.output, **kwargs)


if __name__ == '__main__':
    main()
