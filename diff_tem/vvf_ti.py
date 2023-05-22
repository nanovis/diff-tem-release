import taichi as ti
import torch
from stannum import EmptyTin
from taichi.lang.field import Field
from .utils.helper_functions import check_tensor


@ti.func
def cast_int(x):
    return ti.cast(x, ti.i32)


@ti.kernel
def interpolate_along_x_kernel(grid_point_values: ti.template(),
                               grid_coord_along_axis: ti.template(),
                               point_pos: ti.template(),
                               order_along_x: ti.template(),
                               dy: ti.template(),
                               predecessor_pos: ti.template(),
                               intervals: ti.template(),
                               output_values: ti.template()):
    for point in range(point_pos.shape[0]):
        for ax in range(grid_coord_along_axis.shape[0] - 1):
            interval_start = intervals[point, 0]  # interval_start is floor(predecessor_location)
            interval_end = intervals[point, 1]  # interval_end is floor(point_location)
            is_first = order_along_x[point] == 0
            if not is_first:
                if interval_start <= interval_end:
                    if interval_start <= ax <= interval_end:
                        x = point_pos[point, 0]
                        y = point_pos[point, 1]
                        belong_y_ax = cast_int(y)
                        p_x = predecessor_pos[point, 0]
                        if interval_start == interval_end:
                            # now interval_start == interval_end == ax == floor(x)
                            value = (x - p_x) / dy[None] * grid_point_values[ax, belong_y_ax]
                            output_values[point] += value
                        else:
                            if ax == interval_start:
                                stride = grid_coord_along_axis[ax + 1] - p_x
                                value = stride / dy[None] * grid_point_values[ax, belong_y_ax]
                                output_values[point] += value
                            elif ax == interval_end:
                                stride = x - grid_coord_along_axis[ax]
                                value = stride / dy[None] * grid_point_values[ax, belong_y_ax]
                                output_values[point] += value
                            else:
                                value = (grid_coord_along_axis[ax + 1] - grid_coord_along_axis[ax]) / dy[None] \
                                        * grid_point_values[ax, belong_y_ax]
                                output_values[point] += value
                else:
                    if interval_start <= ax:
                        y = point_pos[point, 1]
                        belong_y_ax = cast_int(y)
                        p_x = predecessor_pos[point, 0]
                        if ax == interval_start:
                            stride = grid_coord_along_axis[ax + 1] - p_x
                            value = stride / dy[None] * grid_point_values[ax, belong_y_ax]
                            output_values[point] += value
                        elif ax < grid_coord_along_axis.shape[0] - 1:
                            value = (grid_coord_along_axis[ax + 1] - grid_coord_along_axis[ax]) / dy[None] \
                                    * grid_point_values[ax, belong_y_ax]
                            output_values[point] += value
                        else:
                            pass


@ti.kernel
def interpolate_along_y_kernel(grid_point_values: ti.template(),
                               grid_coord_along_axis: ti.template(),
                               point_pos: ti.template(),
                               order_along_y: ti.template(),
                               dy: ti.template(),
                               predecessor_pos: ti.template(),
                               intervals: ti.template(),
                               output_values: ti.template()):
    for point in range(point_pos.shape[0]):
        for ax in range(grid_coord_along_axis.shape[0] - 1):
            interval_start = intervals[point, 0]  # interval_start is floor(predecessor_location)
            interval_end = intervals[point, 1]  # interval_end is floor(point_location)
            is_first = order_along_y[point] == 0
            if not is_first:
                if interval_start <= interval_end:
                    if interval_start <= ax <= interval_end:
                        x = point_pos[point, 0]
                        y = point_pos[point, 1]
                        belong_x_ax = cast_int(x)
                        p_y = predecessor_pos[point, 1]
                        if interval_start == interval_end:
                            # now interval_start == interval_end == ax == floor(x)
                            value = (y - p_y) / dy[None] * grid_point_values[belong_x_ax, ax]
                            output_values[point] += value
                        else:
                            if ax == interval_start:
                                stride = grid_coord_along_axis[ax + 1] - p_y
                                value = stride / dy[None] * grid_point_values[belong_x_ax, ax]
                                output_values[point] += value
                            elif ax == interval_end:
                                stride = y - grid_coord_along_axis[ax]
                                value = stride / dy[None] * grid_point_values[belong_x_ax, ax]
                                output_values[point] += value
                            else:
                                value = (grid_coord_along_axis[ax + 1] - grid_coord_along_axis[ax]) / dy[None] \
                                        * grid_point_values[belong_x_ax, ax]
                                output_values[point] += value
                else:
                    if interval_start <= ax:
                        x = point_pos[point, 0]
                        belong_x_ax = cast_int(x)
                        p_y = predecessor_pos[point, 1]
                        if ax == interval_start:
                            stride = grid_coord_along_axis[ax + 1] - p_y
                            value = stride / dy[None] * grid_point_values[belong_x_ax, ax]
                            output_values[point] += value
                        elif ax < grid_coord_along_axis.shape[0] - 1:
                            value = (grid_coord_along_axis[ax + 1] - grid_coord_along_axis[ax]) / dy[None] \
                                    * grid_point_values[belong_x_ax, ax]
                            output_values[point] += value
                        else:
                            pass


def vvf_interpolate(grid_point_values: torch.Tensor,
                    grid_point_values_field: Field,
                    grid_coord_along_axis: torch.Tensor,
                    grid_coord_along_axis_field: Field,
                    point_pos: torch.Tensor,
                    point_pos_field: Field,
                    order_along_axis: torch.Tensor,
                    order_along_axis_field: Field,
                    dy: torch.Tensor,
                    dy_field: Field,
                    predecessor_pos: torch.Tensor,
                    predecessor_pos_field: Field,
                    intervals: torch.Tensor,
                    intervals_field: Field,
                    output_field: Field,
                    axis: str):
    check_tensor(grid_point_values, dimensionality=2)
    if axis == "x":
        kernel = interpolate_along_x_kernel
    elif axis == "y":
        kernel = interpolate_along_y_kernel
    else:
        raise Exception("Invalid input")
    point_num = point_pos.shape[0]
    assert point_num == intervals.shape[0] and point_num == predecessor_pos.shape[0] \
           and point_num == output_field.shape[0]
    tin = EmptyTin(grid_point_values.device, auto_clear_grad=False) \
        .register_kernel(kernel,
                         grid_point_values_field,
                         grid_coord_along_axis_field,
                         point_pos_field,
                         order_along_axis_field,
                         dy_field,
                         predecessor_pos_field,
                         intervals_field,
                         output_field) \
        .register_input_field(grid_point_values_field, complex_dtype=True) \
        .register_input_field(grid_coord_along_axis_field) \
        .register_input_field(point_pos_field) \
        .register_input_field(order_along_axis_field) \
        .register_input_field(dy_field) \
        .register_input_field(predecessor_pos_field) \
        .register_input_field(intervals_field) \
        .register_output_field(output_field, complex_dtype=True) \
        .finish()
    return tin(grid_point_values, grid_coord_along_axis, point_pos, order_along_axis, dy, predecessor_pos, intervals)
