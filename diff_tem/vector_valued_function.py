import logging
from copy import deepcopy
from typing import List

import taichi as ti
from enum import Enum

from taichi.lang.util import to_taichi_type

from .utils.misc import ti_run_on_single_precision
from .vvf_ti import vvf_interpolate
import torch

from .utils.helper_functions import check_tensor

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class VirtualField:
    def __init__(self, tensor: torch.Tensor):
        dtype = tensor.dtype
        if dtype == torch.cdouble or dtype == torch.cfloat:
            dtype = torch.double if dtype == torch.cdouble else torch.float
            self.complex_dtype = True
        else:
            self.complex_dtype = False
        self.dtype = to_taichi_type(dtype)
        self.requires_grad = tensor.requires_grad
        self.shape = tensor.shape

    def concretize(self, fb: ti.FieldsBuilder):
        if self.complex_dtype:
            field = ti.Vector.field(2, dtype=self.dtype, needs_grad=self.requires_grad)
        else:
            field = ti.field(self.dtype, needs_grad=self.requires_grad)

        if self.requires_grad:
            fb.dense(ti.axes(*range(len(self.shape))), self.shape).place(field, field.grad)
        else:
            fb.dense(ti.axes(*range(len(self.shape))), self.shape).place(field)

        return field


class VectorValuedFunction:
    class AxisOrder(Enum):
        XY = "xy"
        YX = "yx"

    def __init__(self, values: torch.Tensor, x_basis: torch.Tensor, y_basis: torch.Tensor, offset: torch.Tensor):
        device = values.device
        check_tensor(x_basis, device=device)
        check_tensor(y_basis, device=device)
        check_tensor(offset, device=device)
        self.device: torch.device = device
        check_tensor(values, dimensionality=2, dtype=torch.cdouble)
        self.values: torch.Tensor = values  # 2D complex tensor
        check_tensor(x_basis, dimensionality=1, shape=(2,))
        self.x_basis: torch.Tensor = x_basis  # 2-dimensional vec
        check_tensor(y_basis, dimensionality=1, shape=(2,))
        self.y_basis: torch.Tensor = y_basis  # 2-dimensional vec
        check_tensor(offset, dimensionality=1, shape=(2,))
        self.offset: torch.Tensor = offset  # 2-dimensional vec
        self._xy_range: torch.Tensor = self._calc_xy_range()  # 4-dimensional vec

    def _calc_xy_range(self):
        s1 = 0.5 * (self.values.shape[1] * torch.abs(self.x_basis[0])
                    + self.values.shape[0] * torch.abs(self.y_basis[0]))
        s2 = 0.5 * (self.values.shape[1] * torch.abs(self.x_basis[1])
                    + self.values.shape[0] * torch.abs(self.y_basis[1]))
        xy_range = torch.tensor([
            self.offset[0] - s1,
            self.offset[0] + s1,
            self.offset[1] - s2,
            self.offset[1] + s2
        ], device=self.device, dtype=torch.double)
        return xy_range

    @property
    def xy_range(self) -> torch.Tensor:
        # CORRESPOND: vecf2d_xyrange
        return torch.clone(self._xy_range)

    def spawn(self):
        """
        :return: a vector valued function with the same geometry but zero values
        """
        spawned_vvf = VectorValuedFunction(torch.zeros_like(self.values),
                                           self.x_basis.clone(),
                                           self.y_basis.clone(),
                                           self.offset.clone())
        return spawned_vvf

    def spawn_with(self, vvf_values: torch.Tensor):
        check_tensor(vvf_values, shape=self.values.shape, dtype=self.values.dtype)
        device = vvf_values.device
        spawned_vvf = VectorValuedFunction(vvf_values,
                                           self.x_basis.clone().to(device),
                                           self.y_basis.clone().to(device),
                                           self.offset.clone().to(device))
        return spawned_vvf

    def to(self, device: torch.device):
        """
        Out-of-place move to a device, creating new copy of self whose data is on the device

        :param device: device that holds data
        :return: a new copy of self
        """
        return VectorValuedFunction(self.values.to(device),
                                    self.x_basis.to(device), self.y_basis.to(device),
                                    self.offset.to(device))

    def to_(self, device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        self.values = self.values.to(device)
        self.x_basis = self.x_basis.to(device)
        self.y_basis = self.y_basis.to(device)
        self.offset = self.offset.to(device)
        self._xy_range = self._xy_range.to(device)

    def merge_standins_(self, standins):
        """
        Add all stand-ins' values to self
        :param standins: standin(s) of self
        """
        if isinstance(standins, VectorValuedFunction):
            self._merge_standins_([standins])
        elif isinstance(standins, list):
            for s in standins:
                assert isinstance(s, VectorValuedFunction), "the instance in list is not VectorValuedFunction"
            self._merge_standins_(standins)
        else:
            raise Exception("Only accept VectorValuedFunction instance or a list of instances")

    def _merge_standins_(self, standins):
        all_values = [self.values, ]
        for s in standins:
            values = s.values
            if values.is_sparse:
                # not used, so values are all zeros
                pass
            else:
                all_values.append(values)
        all_values = torch.stack(all_values, dim=0)
        self.values = all_values.sum(dim=0)

    def add_(self, other):
        """
        Vector valued function add

        :param other: the other VectorValuedFunction
        """

        """
        Coordinates:
        SC: self's coordinate, the indices of self.values reside in SC, N^2
        SCC: self's centralized coordinates, the geometric positions of self.values reside in SCC, R^2
        OC: other's coordinate, the indices of other.values reside in OC, N^2
        OCC: other's centralized coordinate, the geometric positions of other.values reside in OCC, R^2
        WC: world coordinate, VectorValuedFunction.basis reside in WC
        
        Abbreviations:
        SP: self.values' positions
        OP: other.values' positions
        SV: self.values
        OV: other.values
        Example:
            SP_OCC: self.values' positions in OCC
            SV_SP_OCC: self.values at self.values' positions in OCC
            
        TODO: more detailed explanation on the algorithm of VectorValuedFunction.add_()
        """
        assert self.device == other.device
        dimsa = None
        dimsb = None
        c = None
        coeff = None
        for dimsb_ in self.AxisOrder:
            for dimsa_ in self.AxisOrder:
                shear_coeffs = double_shear_coeff(other, self, dimsa_, dimsb_)
                coeff_ = torch.abs(shear_coeffs)[:, 0]
                c_ = torch.maximum(torch.max(coeff_), torch.max(1.0 / coeff_))
                if c is None or c_ < c:
                    c = c_
                    dimsa = dimsa_
                    dimsb = dimsb_
                    coeff = shear_coeffs

        if dimsb == VectorValuedFunction.AxisOrder.XY:
            self_values = self.values.permute(1, 0)
        else:
            self_values = self.values

        self_value_1st_axis_dimension, self_value_2nd_axis_dimension = self_values.shape

        if dimsa == VectorValuedFunction.AxisOrder.XY:
            other_values = other.values.permute(1, 0)
        else:
            other_values = other.values
        other_value_1st_axis_dimension, other_value_2nd_axis_dimension = other_values.shape

        # NG: No contribution to Gradients
        abs_coeff = torch.abs(coeff.detach())  # NG
        zero = torch.zeros(1, device=self.device, dtype=torch.long)
        imin = torch.maximum(zero,
                             torch.floor(0.5 * (other_value_2nd_axis_dimension - 1)
                                         - abs_coeff[1, 1] * 0.5 * (self_value_1st_axis_dimension - 1)
                                         - abs_coeff[1, 0] * 0.5 * (self_value_2nd_axis_dimension - 1)
                                         + coeff[1][2]).long()
                             ).item()  # NG
        imax = torch.minimum(torch.tensor(other_value_2nd_axis_dimension, dtype=torch.long, device=self.device),
                             1 + torch.ceil(0.5 * (other_value_2nd_axis_dimension - 1)
                                            + abs_coeff[1, 1] * 0.5 * (self_value_1st_axis_dimension - 1)
                                            + abs_coeff[1, 0] * 0.5 * (self_value_2nd_axis_dimension - 1)
                                            + coeff[1, 2]).long()
                             ).item()  # NG
        jmin = torch.maximum(zero,
                             torch.floor(0.5 * (self_value_1st_axis_dimension - 1)
                                         - 1.0 / abs_coeff[0, 0] * 0.5 * (other_value_1st_axis_dimension - 1)
                                         - abs_coeff[0, 1] / abs_coeff[0, 0] * 0.5
                                         * (other_value_2nd_axis_dimension - 1)
                                         - coeff[0, 2] / coeff[0, 0]).long()
                             ).item()  # NG
        jmax = torch.minimum(torch.tensor(self_value_1st_axis_dimension, device=self.device, dtype=torch.long),
                             1 + torch.ceil(0.5 * (self_value_1st_axis_dimension - 1)
                                            + 1.0 / abs_coeff[0, 0] * 0.5 * (other_value_1st_axis_dimension - 1)
                                            + abs_coeff[0, 1] / abs_coeff[0, 0] * 0.5
                                            * (other_value_2nd_axis_dimension - 1)
                                            - coeff[0, 2] / coeff[0, 0]).long()
                             ).item()  # NG
        if imin >= imax or jmin >= jmax:
            yield None  # ignore first interpolation
            yield None  # ignore second interpolation
            return
        # FIRST INTERPOLATION
        # print("\n============================ FIRST PASS =====================================\n")

        x_zero_column = torch.zeros(jmax - jmin, dtype=torch.cdouble, device=self.device)
        SV_SP_OCC_new = []
        x0 = zero  # NG
        nx = other_value_1st_axis_dimension  # NG
        ny = jmax - jmin  # NG
        dx = 1.  # NG
        dy = coeff[0, 0]

        is_dy_negative = dy < 0
        if is_dy_negative:
            dy = -dy

        y0 = coeff[0, 2] + 0.5 * (jmin + jmax - self_value_1st_axis_dimension) * coeff[0, 0] \
             + (imin - 0.5 * other_value_2nd_axis_dimension) * coeff[0, 1]
        y0s = torch.arange(imax - imin) * coeff[0, 1] + y0 - 0.5 * dy * ny  # ((imax - imin), )
        ny0s = torch.arange(ny + 1) * dy
        SP_OCC = y0s.reshape(-1, 1) + ny0s.reshape(1, -1)  # (imax - imin, ny + 1), G
        OP_OCC_x = torch.arange(0, nx + 1) + x0 - 0.5 * nx * dx  # (nx + 1), NG
        valid_SP_OCC = []
        valid_SP_OCC_orders = []
        predecessor_pos_list_SP_OCC = []
        inbound_indices = []
        size_inbounds = []
        start_inbound_indices = []
        x_i_not_total_out = []
        k = -1
        for i in range(imax - imin):
            SP_OCC_one_row = SP_OCC[i]
            total_out = (SP_OCC_one_row[-1] < OP_OCC_x[0]) | (SP_OCC_one_row[0] > OP_OCC_x[-1])
            SV_SP_OCC_new.append(x_zero_column)
            if not total_out:
                k = k + 1
                out_bound = (SP_OCC_one_row < OP_OCC_x[0]) | (SP_OCC_one_row > OP_OCC_x[-1])
                in_bound = ~out_bound
                in_bound = in_bound.nonzero()
                if in_bound.shape[0] < SP_OCC_one_row.shape[0]:
                    if in_bound[0, 0] == 0:
                        in_bound = torch.cat([in_bound[:, 0], (in_bound[-1, 0] + 1).view(1)])
                    elif in_bound[-1, 0] == SP_OCC_one_row.shape[0] - 1:
                        in_bound = torch.cat([(in_bound[0, 0] - 1).view(1), in_bound[:, 0]])
                    else:
                        in_bound = torch.cat(
                            [(in_bound[0, 0] - 1).view(1), in_bound[:, 0], (in_bound[-1, 0] + 1).view(1)])

                in_bound = in_bound.squeeze()
                green_point_coords_x = SP_OCC_one_row[in_bound]  # (valid_SP_OCC_num)
                _, sort_idx = torch.sort(green_point_coords_x.detach())
                orders = torch.zeros_like(sort_idx)  # (valid_SP_OCC_num)
                orders[sort_idx] = torch.arange(sort_idx.shape[0])
                predecessor_indices = sort_idx[orders - 1]  # (valid_SP_OCC_num)
                SP_OCC_y = torch.full_like(green_point_coords_x, i)
                valid_SP_OCC_one_row = torch.stack([green_point_coords_x, SP_OCC_y],
                                                   dim=-1)  # (valid_SP_OCC_num, 2)
                predecessor_pos = valid_SP_OCC_one_row[predecessor_indices, :]  # (valid_SP_OCC_num, 2)
                valid_SP_OCC.append(valid_SP_OCC_one_row)
                valid_SP_OCC_orders.append(orders)
                predecessor_pos_list_SP_OCC.append(predecessor_pos)
                inbound_indices.append(in_bound[sort_idx])
                if k == 0:
                    size_inbounds.append(in_bound.shape[0])
                    start_inbound_indices.append(0)
                else:
                    size_inbounds.append(in_bound.shape[0])
                    start_inbound_indices.append(start_inbound_indices[k - 1] + size_inbounds[k - 1])
                x_i_not_total_out.append(i)

        # If other and self do not overlap in X-axis, we do not need to care about overlap in Y-axis
        if len(valid_SP_OCC) == 0:
            yield None  # ignore first interpolation
            yield None  # ignore second interpolation
            return

        valid_SP_OCC = torch.cat(valid_SP_OCC, dim=0)
        predecessor_coords = torch.cat(predecessor_pos_list_SP_OCC, dim=0)
        valid_SP_OCC_orders = torch.cat(valid_SP_OCC_orders, dim=0)
        inbound_indices = torch.cat(inbound_indices, dim=0)
        # temp check
        valid_SP_OCC_num = valid_SP_OCC.shape[0]
        assert valid_SP_OCC_num == predecessor_coords.shape[0] and valid_SP_OCC_num == \
               valid_SP_OCC_orders.shape[0]

        # calculate intervals along x axis
        is_first = valid_SP_OCC_orders == 0
        interval_start_coords = predecessor_coords[:, 0]  # (all_valid_points)
        interval_end_coords = valid_SP_OCC[:, 0]  # (all_valid_points)
        OP_OCC_gt_interval_start = OP_OCC_x.view(1, -1) > interval_start_coords.reshape(-1,
                                                                                        1)  # (all_valid_points, nx+1)
        any_OP_OCC_gt_interval_start = torch.any(OP_OCC_gt_interval_start, dim=1)
        interval_start_idx = torch.max(OP_OCC_gt_interval_start, dim=1)[1] - 1
        interval_start_idx[is_first] = -2
        interval_start_idx[~any_OP_OCC_gt_interval_start] = -2
        OP_OCC_gt_interval_end = OP_OCC_x.view(1, -1) > interval_end_coords.reshape(-1,
                                                                                    1)  # (all_valid_points, nx+1)
        any_OP_OCC_gt_interval_end = torch.any(OP_OCC_gt_interval_end, dim=1)
        interval_end_idx = torch.max(OP_OCC_gt_interval_end, dim=1)[1] - 1
        interval_end_idx[is_first] = -2
        interval_end_idx[~any_OP_OCC_gt_interval_end] = -2
        intervals = torch.stack([interval_start_idx, interval_end_idx], dim=-1)  # (all_valid_points, 2)

        use_single_precision = ti_run_on_single_precision()
        if use_single_precision:
            # cast dtype to fit vulkan backend
            other_values_cf32 = other_values.cfloat()
            OP_OCC_x_f32 = OP_OCC_x.float()
            valid_SP_OCC_f32 = valid_SP_OCC.float()
            dy_f32 = dy.float()
            predecessor_coords_f32 = predecessor_coords.float()
            valid_SP_OCC_orders_i32 = valid_SP_OCC_orders.int()
            intervals_i32 = intervals.int()
            # construct virtual fields to be concretized
            virtual_fields = list(map(VirtualField, [other_values_cf32, OP_OCC_x_f32,
                                                     valid_SP_OCC_f32, valid_SP_OCC_orders_i32,
                                                     dy_f32, predecessor_coords_f32, intervals_i32]))
            output_virtual_field = deepcopy(virtual_fields[0])
            output_virtual_field.shape = (valid_SP_OCC_num,)
            virtual_fields.append(output_virtual_field)
            concrete_fields: List[ti.Field] = yield virtual_fields
            other_values_cf32_field, OP_OCC_x_f32_field, \
            valid_SP_OCC_f32_field, valid_SP_OCC_orders_i32_field, \
            dy_f32_field, predecessor_coords_f32_field, intervals_i32_field, output_field = concrete_fields

            valid_SV_SP_OCC_values = vvf_interpolate(other_values_cf32,
                                                     other_values_cf32_field,
                                                     OP_OCC_x_f32,  # NG
                                                     OP_OCC_x_f32_field,
                                                     valid_SP_OCC_f32,  # G
                                                     valid_SP_OCC_f32_field,
                                                     valid_SP_OCC_orders_i32,  # NG
                                                     valid_SP_OCC_orders_i32_field,
                                                     dy_f32,  # G
                                                     dy_f32_field,
                                                     predecessor_coords_f32,  # G
                                                     predecessor_coords_f32_field,
                                                     intervals_i32,  # NG
                                                     intervals_i32_field,
                                                     output_field,
                                                     "x")
            valid_SV_SP_OCC_values = valid_SV_SP_OCC_values.to(other_values.dtype)
        else:
            # construct virtual fields to be concretized
            virtual_fields = list(map(VirtualField, [other_values, OP_OCC_x,
                                                     valid_SP_OCC, valid_SP_OCC_orders,
                                                     dy, predecessor_coords, intervals]))
            output_virtual_field = deepcopy(virtual_fields[0])
            output_virtual_field.shape = (valid_SP_OCC_num,)
            virtual_fields.append(output_virtual_field)
            concrete_fields: List[ti.Field] = yield virtual_fields
            other_values_field, OP_OCC_x_field, \
            valid_SP_OCC_field, valid_SP_OCC_orders_field, \
            dy_field, predecessor_coords_field, intervals_field, output_field = concrete_fields
            valid_SV_SP_OCC_values = vvf_interpolate(other_values,
                                                     other_values_field,
                                                     OP_OCC_x,  # NG
                                                     OP_OCC_x_field,
                                                     valid_SP_OCC,  # G
                                                     valid_SP_OCC_field,
                                                     valid_SP_OCC_orders,  # NG
                                                     valid_SP_OCC_orders_field,
                                                     dy,  # G
                                                     dy_field,
                                                     predecessor_coords,  # G
                                                     predecessor_coords_field,
                                                     intervals,  # NG
                                                     intervals_field,
                                                     output_field,
                                                     "x")
        _write_log(f'valid_SV_SP_OCC_values X AXIS Grad = {valid_SV_SP_OCC_values.grad_fn}')
        inbound_indices = inbound_indices - 1
        if is_dy_negative:
            inbound_indices = ny - inbound_indices - 1
        cut_inbound_indices = []
        cut_SV_SP_OCC = []
        for i in range(len(start_inbound_indices) - 1):
            cut_inbound_indices.append(inbound_indices[start_inbound_indices[i]:start_inbound_indices[i + 1]])
            cut_SV_SP_OCC.append(
                valid_SV_SP_OCC_values[start_inbound_indices[i]:start_inbound_indices[i + 1]])

        i = len(start_inbound_indices) - 1
        cut_inbound_indices.append(
            inbound_indices[start_inbound_indices[i]: start_inbound_indices[i] + size_inbounds[i]])
        cut_SV_SP_OCC.append(
            valid_SV_SP_OCC_values[start_inbound_indices[i]: start_inbound_indices[i] + size_inbounds[i]])

        for i in range(len(cut_inbound_indices)):
            out_bound_index = (cut_inbound_indices[i] < 0) | (cut_inbound_indices[i] > ny - 1)
            cut_SV_SP_OCC[i] = cut_SV_SP_OCC[i][~out_bound_index]
            cut_inbound_indices[i] = cut_inbound_indices[i][~out_bound_index]

        for i, j in zip(x_i_not_total_out, range(len(cut_inbound_indices))):
            SV_SP_OCC_new[i] = torch.index_add(x_zero_column, 0, cut_inbound_indices[j],
                                               cut_SV_SP_OCC[j])
        SV_SP_OCC_new = torch.stack(SV_SP_OCC_new, dim=-1)

        # SECOND INTERPOLATION
        # print("\n============================ SECOND PASS =====================================\n")
        x0 = torch.tensor(0.5 * (imin + imax - other_value_2nd_axis_dimension))  # NG
        nx = imax - imin  # NG
        ny = self_value_2nd_axis_dimension  # NG
        dx = 1.  # NG
        dy = coeff[1, 0]  # G

        is_dy_negative = dy < 0
        if is_dy_negative:
            dy = -dy

        y0 = coeff[1][2] + (jmin - 0.5 * self_value_1st_axis_dimension) * coeff[1, 1]
        y0s = torch.arange(jmax - jmin) * coeff[1, 1] + y0 - 0.5 * dy * ny  # ((jmax - jmin), )
        ny0s = torch.arange(ny + 1) * dy
        SP_OCC = y0s.reshape(-1, 1) + ny0s.reshape(1, -1)  # (jmax - jmin, ny + 1), G
        OP_OCC_x = torch.arange(0, nx + 1) + x0 - 0.5 * nx * dx  # (nx + 1), NG

        valid_SP_OCC = []
        valid_SP_OCC_orders = []
        predecessor_pos_list_SP_OCC = []
        inbound_indices = []
        size_inbounds = []
        start_inbound_indices = []
        self_tensor_slices = []
        self_slices_not_total_out = []
        k = -1
        for j in range(jmax - jmin):
            SP_OCC_one_column = SP_OCC[j]
            total_out = (SP_OCC_one_column[-1] < OP_OCC_x[0]) | (SP_OCC_one_column[0] > OP_OCC_x[-1])
            self_tensor_slices.append(self_values[j + jmin, :])
            if not total_out:
                k = k + 1
                out_bound = (SP_OCC_one_column < OP_OCC_x[0]) | (SP_OCC_one_column > OP_OCC_x[-1])
                in_bound = ~out_bound
                in_bound = in_bound.nonzero()
                if in_bound.shape[0] < SP_OCC_one_column.shape[0]:
                    if in_bound[0, 0] == 0:
                        in_bound = torch.cat([in_bound[:, 0], (in_bound[-1, 0] + 1).view(1)])
                    elif in_bound[-1, 0] == SP_OCC_one_column.shape[0] - 1:
                        in_bound = torch.cat([(in_bound[0, 0] - 1).view(1), in_bound[:, 0]])
                    else:
                        in_bound = torch.cat(
                            [(in_bound[0, 0] - 1).view(1), in_bound[:, 0], (in_bound[-1, 0] + 1).view(1)])
                in_bound = in_bound.squeeze()
                SP_OCC_y = SP_OCC_one_column[in_bound]  # (valid_SP_OCC_num)
                _, sort_idx = torch.sort(SP_OCC_y.detach())
                orders = torch.zeros_like(sort_idx)  # (valid_SP_OCC_num)
                orders[sort_idx] = torch.arange(sort_idx.shape[0])
                predecessor_indices = sort_idx[orders - 1]  # (valid_SP_OCC_num)
                green_point_coords_x = torch.full_like(SP_OCC_y, j)
                valid_SP_OCC_one_column = torch.stack([green_point_coords_x, SP_OCC_y],
                                                      dim=-1)  # (valid_SP_OCC_num, 2), G
                predecessor_pos = valid_SP_OCC_one_column[predecessor_indices, :]  # (valid_SP_OCC_num, 2)
                valid_SP_OCC.append(valid_SP_OCC_one_column)
                valid_SP_OCC_orders.append(orders)
                predecessor_pos_list_SP_OCC.append(predecessor_pos)
                inbound_indices.append(in_bound[sort_idx])
                if k == 0:
                    size_inbounds.append(in_bound.shape[0])
                    start_inbound_indices.append(0)
                else:
                    size_inbounds.append(in_bound.shape[0])
                    start_inbound_indices.append(start_inbound_indices[k - 1] + size_inbounds[k - 1])
                self_slices_not_total_out.append(j)

        if len(valid_SP_OCC) == 0:
            self_tensor_slices = torch.stack(self_tensor_slices, dim=0)
            _write_log(f'self_tensor_slices Grad = {self_tensor_slices.grad_fn}')
            self_values[jmin:jmax, :] = self_tensor_slices
            yield None  # ignore second interpolation
            return

        valid_SP_OCC = torch.cat(valid_SP_OCC, dim=0)
        predecessor_coords = torch.cat(predecessor_pos_list_SP_OCC, dim=0)
        valid_SP_OCC_orders = torch.cat(valid_SP_OCC_orders, dim=0)
        inbound_indices = torch.cat(inbound_indices, dim=0)

        # temp check
        valid_SP_OCC_num = valid_SP_OCC.shape[0]
        assert valid_SP_OCC_num == predecessor_coords.shape[0] and valid_SP_OCC_num == \
               valid_SP_OCC_orders.shape[0]

        # calculate intervals along y axis
        is_first = valid_SP_OCC_orders == 0
        interval_start_coords = predecessor_coords[:, 1]  # (all_valid_points)
        interval_end_coords = valid_SP_OCC[:, 1]  # (all_valid_points)
        OP_OCC_gt_interval_start = OP_OCC_x.view(1, -1) > interval_start_coords.reshape(-1,
                                                                                        1)  # (all_valid_points, ny+1)
        any_OP_OCC_gt_interval_start = torch.any(OP_OCC_gt_interval_start, dim=1)
        interval_start_idx = torch.max(OP_OCC_gt_interval_start, dim=1)[1] - 1
        interval_start_idx[is_first] = -2
        interval_start_idx[~any_OP_OCC_gt_interval_start] = -2
        OP_OCC_gt_interval_end = OP_OCC_x.view(1, -1) > interval_end_coords.reshape(-1,
                                                                                    1)  # (all_valid_points, ny+1)
        any_OP_OCC_gt_interval_end = torch.any(OP_OCC_gt_interval_end, dim=1)
        interval_end_idx = torch.max(OP_OCC_gt_interval_end, dim=1)[1] - 1
        interval_end_idx[is_first] = -2
        interval_end_idx[~any_OP_OCC_gt_interval_end] = -2
        intervals = torch.stack([interval_start_idx, interval_end_idx], dim=-1)  # (all_valid_points, 2)

        if use_single_precision:
            # cast dtype to fit vulkan backend
            SV_SP_OCC_new_cf32 = SV_SP_OCC_new.cfloat()
            OP_OCC_x_f32 = OP_OCC_x.float()
            valid_SP_OCC_f32 = valid_SP_OCC.float()
            dy_f32 = dy.float()
            predecessor_coords_f32 = predecessor_coords.float()
            valid_SP_OCC_orders_i32 = valid_SP_OCC_orders.int()
            intervals_i32 = intervals.int()

            # construct virtual fields to be concretized
            virtual_fields = list(map(VirtualField, [SV_SP_OCC_new_cf32, OP_OCC_x_f32,
                                                     valid_SP_OCC_f32, valid_SP_OCC_orders_i32,
                                                     dy_f32, predecessor_coords_f32, intervals_i32]))
            output_virtual_field = deepcopy(virtual_fields[0])
            output_virtual_field.shape = (valid_SP_OCC_num,)
            virtual_fields.append(output_virtual_field)

            concrete_fields: List[ti.Field] = yield virtual_fields
            SV_SP_OCC_new_cf32_field, OP_OCC_x_f32_field, \
            valid_SP_OCC_f32_field, valid_SP_OCC_orders_i32_field, \
            dy_f32_field, predecessor_coords_f32_field, intervals_i32_field, output_field = concrete_fields

            valid_SV_SP_OCC_values = vvf_interpolate(SV_SP_OCC_new_cf32,  # NG
                                                     SV_SP_OCC_new_cf32_field,
                                                     OP_OCC_x_f32,  # NG
                                                     OP_OCC_x_f32_field,
                                                     valid_SP_OCC_f32,  # G
                                                     valid_SP_OCC_f32_field,
                                                     valid_SP_OCC_orders_i32,  # NG
                                                     valid_SP_OCC_orders_i32_field,
                                                     dy_f32,  # G
                                                     dy_f32_field,
                                                     predecessor_coords_f32,  # G
                                                     predecessor_coords_f32_field,
                                                     intervals_i32,  # NG
                                                     intervals_i32_field,
                                                     output_field,
                                                     "y")
            valid_SV_SP_OCC_values = valid_SV_SP_OCC_values.to(SV_SP_OCC_new.dtype)
        else:
            # construct virtual fields to be concretized
            virtual_fields = list(map(VirtualField, [SV_SP_OCC_new, OP_OCC_x,
                                                     valid_SP_OCC, valid_SP_OCC_orders,
                                                     dy, predecessor_coords, intervals]))
            output_virtual_field = deepcopy(virtual_fields[0])
            output_virtual_field.shape = (valid_SP_OCC_num,)
            virtual_fields.append(output_virtual_field)

            concrete_fields: List[ti.Field] = yield virtual_fields
            SV_SP_OCC_new_field, OP_OCC_x_field, \
            valid_SP_OCC_field, valid_SP_OCC_orders_field, \
            dy_field, predecessor_coords_field, intervals_field, output_field = concrete_fields
            valid_SV_SP_OCC_values = vvf_interpolate(SV_SP_OCC_new,  # NG
                                                     SV_SP_OCC_new_field,
                                                     OP_OCC_x,  # NG
                                                     OP_OCC_x_field,
                                                     valid_SP_OCC,  # G
                                                     valid_SP_OCC_field,
                                                     valid_SP_OCC_orders,  # NG
                                                     valid_SP_OCC_orders_field,
                                                     dy,  # G
                                                     dy_field,
                                                     predecessor_coords,  # G
                                                     predecessor_coords_field,
                                                     intervals,  # NG
                                                     intervals_field,
                                                     output_field,
                                                     "y")
        _write_log(f'valid_SV_SP_OCC_values Y AXIS Grad = {valid_SV_SP_OCC_values.grad_fn}')
        inbound_indices = inbound_indices - 1
        if is_dy_negative:
            inbound_indices = ny - inbound_indices - 1
        cut_inbound_indices = []
        cut_SV_SP_OCC = []
        for i in range(len(start_inbound_indices) - 1):
            cut_inbound_indices.append(inbound_indices[start_inbound_indices[i]:start_inbound_indices[i + 1]])
            cut_SV_SP_OCC.append(
                valid_SV_SP_OCC_values[start_inbound_indices[i]:start_inbound_indices[i + 1]])

        i = len(start_inbound_indices) - 1
        cut_inbound_indices.append(
            inbound_indices[start_inbound_indices[i]: start_inbound_indices[i] + size_inbounds[i]])
        cut_SV_SP_OCC.append(
            valid_SV_SP_OCC_values[start_inbound_indices[i]: start_inbound_indices[i] + size_inbounds[i]])

        for i in range(len(cut_inbound_indices)):
            out_bound_index = (cut_inbound_indices[i] < 0) | (cut_inbound_indices[i] > ny - 1)
            cut_SV_SP_OCC[i] = cut_SV_SP_OCC[i][~out_bound_index]
            cut_inbound_indices[i] = cut_inbound_indices[i][~out_bound_index]

        for i, j in zip(self_slices_not_total_out, range(len(cut_inbound_indices))):
            self_tensor_slices[i] = torch.index_add(self_values[i + jmin, :], 0, cut_inbound_indices[j],
                                                    cut_SV_SP_OCC[j])

        self_tensor_slices = torch.stack(self_tensor_slices, dim=0)
        _write_log(f'self_tensor_slices Grad = {self_tensor_slices.grad_fn}')
        self_values[jmin:jmax, :] = self_tensor_slices
        _write_log(f'self.values Grad = {self.values.grad_fn}')


def double_shear_coeff(a: VectorValuedFunction,
                       b: VectorValuedFunction,
                       dimsa: VectorValuedFunction.AxisOrder,
                       dimsb: VectorValuedFunction.AxisOrder) -> torch.Tensor:
    if dimsa == VectorValuedFunction.AxisOrder.XY:
        matrix_u = torch.stack([a.x_basis, a.y_basis], dim=1)
    else:
        matrix_u = torch.stack([a.y_basis, a.x_basis], dim=1)

    if dimsb == VectorValuedFunction.AxisOrder.XY:
        matrix_v = torch.stack([b.x_basis, b.y_basis, b.offset - a.offset], dim=1)
    else:
        matrix_v = torch.stack([b.y_basis, b.x_basis, b.offset - a.offset], dim=1)

    # CORRESPOND: solve_2x2_matrix
    w = torch.linalg.solve(matrix_u, matrix_v)  # solve 2x2 matrix
    coeff_matrix = torch.ones_like(matrix_v)
    c01 = w[0, 1] / w[1, 1]
    coeff_matrix[0, 0] = w[0, 0] - c01 * w[1, 0]
    coeff_matrix[0, 1] = c01
    coeff_matrix[0, 2] = w[0, 2] - c01 * w[1, 2]
    coeff_matrix[1, 0] = w[1, 1]
    coeff_matrix[1, 1] = w[1, 0]
    coeff_matrix[1, 2] = w[1, 2]
    return coeff_matrix
