from math import sqrt

from shapely.geometry import Polygon
from shapely.ops import unary_union

try:
    import ocl
except ImportError:
    try:
        import opencamlib as ocl
    except ImportError:
        pass


from ..constants import (
    OCL_SCALE,
)
from .async_utils import progress_async
from ..chunk_builder import CamPathChunk
from .chunk_utils import limit_chunks, sort_chunks
from .logging_utils import log
from .ocl_utils import get_oclSTL
from .parent_utils import parent_child_distance
from .shapely_utils import shapely_to_chunks


def oclWaterlineLayerHeights(operation):
    """Generate a list of waterline layer heights for a given operation.

    This function calculates the heights of waterline layers based on the
    specified parameters of the operation. It starts from the maximum height
    and decrements by a specified step until it reaches the minimum height.
    The resulting list of heights can be used for further processing in
    operations that require layered depth information.

    Args:
        operation (object): An object containing the properties `minz`,
            `maxz`, and `stepdown` which define the
            minimum height, maximum height, and step size
            for layer generation, respectively.

    Returns:
        list: A list of waterline layer heights from maximum to minimum.
    """
    layers = []
    l_last = operation.min_z
    l_step = operation.stepdown
    l_first = operation.max_z - l_step
    l_depth = l_first
    while l_depth > (l_last + 0.0000001):
        layers.append(l_depth)
        l_depth -= l_step
    layers.append(l_last)
    return layers


async def oclGetWaterline(operation, chunks):
    """Generate waterline paths for a given machining operation.

    This function calculates the waterline paths based on the provided
    machining operation and its parameters. It determines the appropriate
    cutter type and dimensions, sets up the waterline object with the
    corresponding STL file, and processes each layer to generate the
    machining paths. The resulting paths are stored in the provided chunks
    list. The function also handles different cutter types, including end
    mills, ball nose cutters, and V-carve cutters.

    Args:
        operation (Operation): An object representing the machining operation,
            containing details such as cutter type, diameter, and minimum Z height.
        chunks (list): A list that will be populated with the generated
            machining path chunks.
    """

    layers = oclWaterlineLayerHeights(operation)
    oclSTL = get_oclSTL(operation)
    op_cutter_type = operation.cutter_type
    op_cutter_diameter = operation.cutter_diameter
    op_minz = operation.min_z

    if op_cutter_type == "VCARVE":
        op_cutter_tip_angle = operation["cutter_tip_angle"]

    cutter = None
    # TODO: automatically determine necessary cutter length depending on object size
    cutter_length = 150

    if op_cutter_type == "END":
        cutter = ocl.CylCutter((op_cutter_diameter + operation.skin * 2) * 1000, cutter_length)
    elif op_cutter_type == "BALLNOSE":
        cutter = ocl.BallCutter((op_cutter_diameter + operation.skin * 2) * 1000, cutter_length)
    elif op_cutter_type == "VCARVE":
        cutter = ocl.ConeCutter(
            (op_cutter_diameter + operation.skin * 2) * 1000, op_cutter_tip_angle, cutter_length
        )
    else:
        log.info(f"Cutter Unsupported: {op_cutter_type}\n")
        quit()

    waterline = ocl.Waterline()
    waterline.setSTL(oclSTL)
    waterline.setCutter(cutter)
    waterline.setSampling(0.1)  # TODO: add sampling setting to UI
    last_pos = [0, 0, 0]

    do_fill = getattr(operation, "waterline_fill", False)
    last_slice = Polygon()

    for count, height in enumerate(layers):
        layer_chunks = []
        await progress_async("Waterline", int((100 * count) / len(layers)))
        waterline.reset()
        waterline.setZ(height * OCL_SCALE)
        waterline.run2()
        wl_loops = waterline.getLoops()

        layer_polys = []
        for l in wl_loops:
            inpoints = []

            for p in l:
                inpoints.append((p.x / OCL_SCALE, p.y / OCL_SCALE, p.z / OCL_SCALE))

            inpoints.append(inpoints[0])
            chunk = CamPathChunk(inpoints=inpoints)
            chunk.closed = True
            layer_chunks.append(chunk)

            if do_fill:
                pts2d = [(pt[0], pt[1]) for pt in inpoints]
                if len(pts2d) >= 4:
                    try:
                        poly = Polygon(pts2d)
                        if poly.is_valid:
                            layer_polys.append(poly)
                    except Exception:
                        pass

        if do_fill:
            current_poly = (
                unary_union([p.buffer(0) for p in layer_polys]) if layer_polys else Polygon()
            )
            last_fill = layer_chunks

            # Fill between slices: clear the flat area newly exposed at this Z
            # Layers iterate top-down so current_poly grows as Z decreases
            if not last_slice.is_empty:
                if getattr(operation, "inverse", False):
                    restpoly = last_slice.difference(current_poly)
                else:
                    restpoly = current_poly.difference(last_slice)
                last_fill = await _fill_polygon(
                    restpoly, height, operation, layer_chunks, last_fill
                )

            # Ambient fill: clear stock area outside the model at this Z
            ambient = getattr(operation, "ambient", None)
            if ambient is not None:
                if getattr(operation, "inverse", False) and current_poly.is_empty:
                    restpoly = ambient.difference(last_slice)
                else:
                    restpoly = ambient.difference(current_poly)
                last_fill = await _fill_polygon(
                    restpoly, height, operation, layer_chunks, last_fill
                )

            last_slice = current_poly

        # sort chunks so that ordering is stable
        chunks.extend(await sort_chunks(layer_chunks, operation, last_pos=last_pos))

        if len(chunks) > 0:
            last_pos = chunks[-1].get_point(-1)


async def _fill_polygon(restpoly, z, operation, layer_chunks, last_fill):
    """Fill a polygon area with concentric inward-offset passes."""
    restpoly = restpoly.buffer(
        -operation.distance_between_paths,
        resolution=operation.optimisation.circle_detail,
    )
    i = 0
    if not restpoly.is_empty:
        max_iters = max(
            100,
            int(sqrt(max(restpoly.area, 0)) / operation.distance_between_paths) + 500,
        )
    else:
        max_iters = 0

    while not restpoly.is_empty and i < max_iters:
        if i % 50 == 0:
            await progress_async("Waterline Fill", i)
        nchunks = shapely_to_chunks(restpoly, z)
        nchunks = limit_chunks(nchunks, operation, force=True)
        layer_chunks.extend(nchunks)
        parent_child_distance(last_fill, nchunks, operation)
        last_fill = nchunks
        restpoly = restpoly.buffer(
            -operation.distance_between_paths,
            resolution=operation.optimisation.circle_detail,
        )
        i += 1
    return last_fill
