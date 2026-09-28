"""Exact periodic square-grid symmetries in physical field units."""

import torch


def periodic_translate_rot90(values, shape, vector_ids, *, shifts=None, rotations=None):
    """Rotate y/x array axes and remap vector components like the source data.

    For one np.rot90-equivalent turn, (vx,vy) becomes (vy,-vx).
    Normalization and observation pooling must happen after this operation.
    """
    height, width = shape
    if height != width:
        raise ValueError("periodic_translate_rot90 requires a square grid")
    if shifts is None:
        shifts = (int(torch.randint(height, (1,))), int(torch.randint(width, (1,))))
    if rotations is None:
        rotations = int(torch.randint(4, (1,)))
    grid = torch.roll(values.reshape(height, width, -1), shifts, dims=(0, 1))
    grid = torch.rot90(grid, rotations, dims=(0, 1)).clone()
    vx, vy = vector_ids
    first, second = grid[..., vx].clone(), grid[..., vy].clone()
    pairs = ((first, second), (second, -first), (-first, -second), (-second, first))
    grid[..., vx], grid[..., vy] = pairs[rotations % 4]
    return grid.reshape_as(values).contiguous()
