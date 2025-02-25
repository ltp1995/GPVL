# Visvalingam-Wyatt

A Python implementation of the Visvalingam-Wyatt line simplification algorithm.

This implementation is due to [Eliot Hallmark](https://github.com/Permafacture/Py-Visvalingam-Whyatt/). This release simply packages it as a Python module.

## Use

```python
>>> import visvalingamwyatt as vw
>>> points = [(1, 2), (2, 3), (3, 4), ...]
>>> vw.simplify(points)
[(1, 2), (3, 4), ...]
```

Points may be any `Sequence`-like object that (`list`, `tuple`, a custom class that exposes an `__iter__` method).

Test different methods and thresholds:
```python
simplifier = vw.Simplifier(points)

# Simplify by percentage of points to keep
simplifier.simplify(ratio=0.5)

# Simplify by giving number of points to keep
simplifier.simplify(number=1000)

# Simplify by giving an area threshold (in the units of the data)
simplifier.simplify(threshold=0.01)
```

Shorthands for working with geodata:

````python
import visvalingamwyatt as vw

feature = {
    "properties": {"foo": "bar"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [...]
    }
}

# returns a copy of the geometry, simplified (keeping 90% of points)
vw.simplify_geometry(feature['geometry'], ratio=0.90)

# returns a copy of the feature, simplified (using an area threshold)
vw.simplify_feature(feature, threshold=0.90)
````

The command line tool `vwsimplify` is available to simplify GeoJSON files:

````
# Simplify using a ratio of points
vwsimplify --ratio 0.90 in.geojson -o simple.geojson

# Simplify using the number of points to keep
vwsimplify --number 1000 in.geojson -o simple.geojson

# Simplify using a minimum area
vwsimplify --threshold 0.001 in.geojson -o simple.geojson
````

Install [Fiona](https://github.com/Toblerity/Fiona) for the additional ability to simplify any geodata layer.

## License

MIT
