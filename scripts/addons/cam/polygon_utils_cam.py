"""BlenderCAM 'polygon_utils_cam.py' © 2012 Vilem Novak

Functions to handle shapely operations and conversions - curve, coords, polygon
"""

from math import pi

import shapely
from shapely.geometry import polygon as spolygon
from shapely import geometry as sgeometry

from mathutils import Euler, Vector
try:
    import curve_simplify
except ImportError:
    pass


SHAPELY = True


def Circle(r, np):
    c = []
    v = Vector((r, 0, 0))
    e = Euler((0, 0, 2.0 * pi / np))
    for a in range(0, np):
        c.append((v.x, v.y))
        v.rotate(e)

    p = spolygon.Polygon(c)
    return p


def shapelyRemoveDoubles(p, optimize_threshold):
    optimize_threshold *= 0.000001

    soptions = ['distance', 'distance', 0.0, 5, optimize_threshold, 5, optimize_threshold]
    for ci, c in enumerate(p.boundary):  # in range(0,len(p)):

        veclist = []
        for v in c:
            veclist.append(Vector((v[0], v[1])))
        s = curve_simplify.simplify_RDP(veclist, soptions)
        nc = []
        for i in range(0, len(s)):
            nc.append(c[s[i]])

        if len(nc) > 2:
            pnew.addContour(nc, p.isHole(ci))
        else:
            pnew.addContour(p[ci], p.isHole(ci))
    return pnew


def shapelyToMultipolygon(anydata):
    if anydata.geom_type == 'MultiPolygon':
        return anydata
    elif anydata.geom_type == 'Polygon':
        if not anydata.is_empty:
            return shapely.geometry.MultiPolygon([anydata])
        else:
            return sgeometry.MultiPolygon()
    else:
        print(anydata.geom_type, 'Shapely Conversion Aborted')
        return sgeometry.MultiPolygon()


def shapelyToCoords(anydata):
    p = anydata
    seq = []
    # print(p.type)
    # print(p.geom_type)
    if p.is_empty:
        return seq
    elif p.geom_type == 'Polygon':

        # print('polygon')
        clen = len(p.exterior.coords)
        # seq = sgeometry.asMultiLineString(p)
        seq = [p.exterior.coords]
        # print(len(p.interiors))
        for interior in p.interiors:
            seq.append(interior.coords)
    elif p.geom_type == 'MultiPolygon':
        clen = 0
        seq = []
        for sp in p.geoms:
            clen += len(sp.exterior.coords)
            seq.append(sp.exterior.coords)
            for interior in sp.interiors:
                seq.append(interior.coords)

    elif p.geom_type == 'MultiLineString':
        seq = []
        for linestring in p.geoms:
            seq.append(linestring.coords)
    elif p.geom_type == 'LineString':
        seq = []
        seq.append(p.coords)

    elif p.geom_type == 'MultiPoint':
        return
    elif p.geom_type == 'GeometryCollection':
        # print(dir(p))
        # print(p.geometryType, p.geom_type)
        clen = 0
        seq = []
        # print(p.boundary.coordsd)
        for sp in p.geoms:  # TODO
            clen += len(sp.exterior.coords)
            seq.append(sp.exterior.coords)
            for interior in sp.interiors:
                seq.extend(interior.coords)

    return seq


def shapelyToCurve(name, p, z):
    import bpy
    import bmesh
    from bpy_extras import object_utils
    verts = []
    edges = []
    vi = 0
    ci = 0
    # for c in p.exterior.coords:

    # print(p.type)
    seq = shapelyToCoords(p)
    w = 1  # weight

    curvedata = bpy.data.curves.new(name=name, type='CURVE')
    curvedata.dimensions = '3D'

    objectdata = bpy.data.objects.new(name, curvedata)
    objectdata.location = (0, 0, 0)  # object origin
    bpy.context.collection.objects.link(objectdata)

    for c in seq:
        polyline = curvedata.splines.new('POLY')
        polyline.points.add(len(c) - 1)
        for num in range(len(c)):
            x, y = c[num][0], c[num][1]
            polyline.points[num].co = (x, y, z, w)

    bpy.context.view_layer.objects.active = objectdata
    objectdata.select_set(state=True)

    for c in objectdata.data.splines:
        c.use_cyclic_u = True

    return objectdata  # bpy.context.active_object
