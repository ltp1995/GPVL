# -*- coding: utf-8 -*-
# This file is part of visvalingamwyatt.
# https://github.com/fitnr/visvalingamwyatt
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2015, fitnr <contact@fakeisthenewreal.org>
"""visvalingamwyatt module tests"""
import json
import os
import unittest
from collections import namedtuple

import numpy as np

import visvalingamwyatt as vw
from visvalingamwyatt import __main__ as cli


class TestVW(unittest.TestCase):
    def setUp(self):
        self.samplefile = os.path.join(os.path.dirname(__file__), 'data', 'sample.json')
        with open(self.samplefile) as f:
            self.fixture = json.load(f).get('features')[0]

    def standard(self, **kwargs):
        result = vw.simplify_feature(self.fixture, **kwargs)

        self.assertIn('geometry', result)
        self.assertIn('properties', result)
        self.assertEqual(result['properties'], self.fixture['properties'])
        self.assertEqual(self.fixture['geometry']['type'], result['geometry']['type'])
        self.assertEqual(
            self.fixture['geometry']['coordinates'][0],
            result['geometry']['coordinates'][0],
        )

        self.assertGreater(
            len(self.fixture['geometry']['coordinates']),
            len(result['geometry']['coordinates']),
        )

        return result

    def testSimplifyFeature(self):
        self.standard()

    def testSimplifyFeatureThreshold(self):
        self.standard(threshold=0.1)

    def testSimplifyFeatureRatio(self):
        result = self.standard(ratio=0.1)

        b = vw.simplify_feature(self.fixture, ratio=0.90)
        assert len(b['geometry']['coordinates']) > len(
            result['geometry']['coordinates']
        )
        for i, j in zip(range(1, 9), range(2, 10)):
            r = vw.simplify_feature(self.fixture, ratio=float(i) / 10)
            s = vw.simplify_feature(self.fixture, ratio=float(j) / 10)
            assert len(r['geometry']['coordinates']) <= len(
                s['geometry']['coordinates']
            )

    def testSimplifyFeatureNumber(self):
        result = self.standard(number=10)
        self.assertEqual(len(result['geometry']['coordinates']), 10)

    def test3dCoords(self):
        coordinates = [
            [0.0, 0.0, 0.0],
            [1.1, 0, 1],
            [2.1, 3, 0],
            [4.1, 5, 10],
            [1.1, 2, 0],
            [5.1, 2, 0],
        ]
        a = vw.simplify(coordinates)
        self.assertEqual(a[0], [0, 0, 0])
        self.assertLessEqual(len(a), len(coordinates))

    def testSimplifyTupleLike(self):
        Point = namedtuple("Point", ("x", "y"))

        # coordinates are in the shape
        #
        #     c
        #   b  d
        #  a    e
        #
        # so b and d are eliminated
        a, b, c, d, e = Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 1), Point(4, 0)
        inp = [a, b, c, d, e]
        expected_output = np.array([a, c, e])

        actual_output = vw.simplify(inp, threshold=0.001)
        self.assertTrue(np.array_equal(actual_output, expected_output))

    def testSimplifyIntegerCoords(self):
        # coordinates are in the shape
        #
        #     c
        #   b  d
        #  a    e
        #
        # so b and d are eliminated
        a, b, c, d, e = (0, 0), (1, 1), (2, 2), (3, 1), (4, 0)
        inp = [a, b, c, d, e]
        expected_output = np.array([a, c, e])

        actual_output = vw.simplify(inp, threshold=0.001)
        self.assertTrue(np.array_equal(actual_output, expected_output))

    def testSimplifyClosedFeature(self):
        '''When simplifying geometries with closed rings (Polygons and MultiPolygons),
        the first and last points in each ring should remain the same'''
        test_ring = [
            [121.20803833007811, 24.75431413309125],
            [121.1846923828125, 24.746831298412058],
            [121.1517333984375, 24.74059525872194],
            [121.14486694335936, 24.729369599118222],
            [121.12152099609375, 24.693191139677126],
            [121.13525390625, 24.66449040712424],
            [121.10504150390625, 24.66449040712424],
            [121.10092163085936, 24.645768980151793],
            [121.0748291015625, 24.615808859044243],
            [121.09405517578125, 24.577099744289427],
            [121.12564086914062, 24.533381526147682],
            [121.14624023437499, 24.515889973088104],
            [121.19018554687499, 24.528384188171866],
            [121.19430541992186, 24.57959746772822],
            [121.23687744140624, 24.587090339209634],
            [121.24099731445311, 24.552119771544227],
            [121.2451171875, 24.525885444592642],
            [121.30279541015624, 24.55087064225044],
            [121.27258300781251, 24.58958786341259],
            [121.26708984374999, 24.623299562653035],
            [121.32614135742188, 24.62579636412304],
            [121.34674072265624, 24.602074737077242],
            [121.36871337890625, 24.580846310771612],
            [121.40853881835936, 24.653257887871963],
            [121.40853881835936, 24.724380091871726],
            [121.37283325195312, 24.716895455859337],
            [121.3604736328125, 24.693191139677126],
            [121.343994140625, 24.69942955501979],
            [121.32888793945312, 24.728122241065808],
            [121.3714599609375, 24.743089712134605],
            [121.37695312499999, 24.77177232822881],
            [121.35635375976562, 24.792968265314457],
            [121.32476806640625, 24.807927923059236],
            [121.29730224609375, 24.844072974931866],
            [121.24923706054688, 24.849057671305268],
            [121.24786376953125, 24.816653556469955],
            [121.27944946289062, 24.79047481357294],
            [121.30142211914061, 24.761796517185815],
            [121.27258300781251, 24.73311159823193],
            [121.25335693359374, 24.708162811665265],
            [121.20391845703125, 24.703172454280217],
            [121.19979858398438, 24.731864277701714],
            [121.20803833007811, 24.75431413309125],
        ]
        multipolygon = {"type": "MultiPolygon", "coordinates": [[test_ring]]}
        number = vw.simplify_geometry(multipolygon, number=10)
        self.assertEqual(
            number['coordinates'][0][0][0], number['coordinates'][0][0][-1]
        )

        ratio = vw.simplify_geometry(multipolygon, ratio=0.3)
        self.assertEqual(ratio['coordinates'][0][0][0], ratio['coordinates'][0][0][-1])

        thres = vw.simplify_geometry(multipolygon, threshold=0.01)
        self.assertEqual(thres['coordinates'][0][0][0], thres['coordinates'][0][0][-1])

        number = vw.simplify_geometry(multipolygon, number=10)
        self.assertEqual(
            number['coordinates'][0][0][0], number['coordinates'][0][0][-1]
        )

        ratio = vw.simplify_geometry(multipolygon, ratio=0.3)
        self.assertEqual(ratio['coordinates'][0][0][0], ratio['coordinates'][0][0][-1])

        thres = vw.simplify_geometry(multipolygon, threshold=0.01)
        self.assertEqual(thres['coordinates'][0][0][0], thres['coordinates'][0][0][-1])

    def testCli(self):
        pass

    def testSimplify(self):
        '''Use the command-line function to simplify the sample data.'''
        try:
            output = 'tmp.json'
            cli.simplify(self.samplefile, output, number=9)

            self.assertTrue(os.path.exists(output))

            with open('tmp.json', 'r') as f:
                result = json.load(f)
                coords = result['features'][0]['geometry']['coordinates']
                self.assertEqual(len(coords), 9)

        finally:
            os.remove(output)

if __name__ == '__main__':
    unittest.main()
