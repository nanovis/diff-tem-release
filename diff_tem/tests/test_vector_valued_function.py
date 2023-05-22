import time
from torch.multiprocessing import Pool
import torch
from diff_tem.vector_valued_function import VectorValuedFunction
import taichi as ti
import logging

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.info(f"{_logger.name}:\n    {logging_string}")
    else:
        _logger.info(f"{_logger.name}:  {strings[0]}")


def test_case_0_with_grad():
    ti.init(ti.vulkan, debug=True)
    values = torch.ones((14, 14)).double()
    values = torch.complex(values, values)
    pf1 = VectorValuedFunction(values,
                               torch.tensor([15.0, 0.0], requires_grad=True),
                               torch.tensor([0.0, 15.0], requires_grad=True),
                               torch.tensor([0.0, 0.], requires_grad=True))

    values = torch.zeros((14, 14)).double()
    values = torch.complex(values, values)
    pf2 = VectorValuedFunction(values,
                               torch.tensor([15.0, 0.0], requires_grad=True),
                               torch.tensor([0.0, 15.0], requires_grad=True),
                               torch.tensor([0.0, 0.], requires_grad=True))

    task = pf2.add_(pf1)
    virtual_fields = task.send(None)
    concrete_fields = []
    if virtual_fields is not None:
        field_builder = ti.FieldsBuilder()
        for vf in virtual_fields:
            concrete_fields.append(vf.concretize(field_builder))
        snode1 = field_builder.finalize()
        for cf, vf in zip(concrete_fields, virtual_fields):
            if vf.requires_grad:
                cf.grad.fill(0)

    virtual_fields = task.send(concrete_fields)
    concrete_fields = []
    if virtual_fields is not None:
        field_builder = ti.FieldsBuilder()
        for vf in virtual_fields:
            concrete_fields.append(vf.concretize(field_builder))
        snode2 = field_builder.finalize()
        for cf, vf in zip(concrete_fields, virtual_fields):
            if vf.requires_grad:
                cf.grad.fill(0)

    try:
        task.send(concrete_fields)
    except StopIteration:
        _write_log("finish")
        pass
    # ============================

    values = torch.load('./wf_intens_not_add.pt')
    pf = VectorValuedFunction(values,
                              torch.tensor([15000.0, 0.0], requires_grad=True),
                              torch.tensor([0.0, 15000.0], requires_grad=True),
                              torch.tensor([0.0, 0.], requires_grad=True))

    values = torch.load('./self_count_not_add.pt')
    wf_phase = VectorValuedFunction(values,
                                    torch.tensor([15000.0, 0.0], requires_grad=True),
                                    torch.tensor([0.0, 15000.0], requires_grad=True),
                                    torch.tensor([0.0, 0.], requires_grad=True))

    task = wf_phase.add_(pf)

    virtual_fields = task.send(None)
    concrete_fields = []
    if virtual_fields is not None:
        field_builder = ti.FieldsBuilder()
        for vf in virtual_fields:
            concrete_fields.append(vf.concretize(field_builder))
        snode1 = field_builder.finalize()
        for cf, vf in zip(concrete_fields, virtual_fields):
            if vf.requires_grad:
                cf.grad.fill(0)

    virtual_fields = task.send(concrete_fields)
    concrete_fields = []
    if virtual_fields is not None:
        field_builder = ti.FieldsBuilder()
        for vf in virtual_fields:
            concrete_fields.append(vf.concretize(field_builder))
        snode2 = field_builder.finalize()
        for cf, vf in zip(concrete_fields, virtual_fields):
            if vf.requires_grad:
                cf.grad.fill(0)

    try:
        task.send(concrete_fields)
    except StopIteration:
        _write_log("finish")
        pass

    # ind = wf_phase.values.real > 0
    # _write_log(wf_phase.values[ind])
    # ind = ind.nonzero()
    # _write_log(ind.shape)
    # _write_log(ind)

    # correct_res = torch.load('./self_count_bf.pt')
    # ind = torch.allclose(correct_res, wf_phase.values)
    # _write_log(f'CORRECT OR NOT: {ind}')
    # torch.save(wf_phase.values, f'test_self_count_vulkan.pt')

    # l = torch.view_as_real(wf_phase.values).sum()
    # l.backward()
    ti.reset()
    ti.init(ti.vulkan, debug=True)

    values = torch.load('./wf_intens_not_add.pt')
    pf3 = VectorValuedFunction(values,
                               torch.tensor([15000.0, 0.0], requires_grad=True),
                               torch.tensor([0.0, 15000.0], requires_grad=True),
                               torch.tensor([0.0, 0.], requires_grad=True))

    values = torch.load('./self_count_not_add.pt')
    wf_phase2 = VectorValuedFunction(values,
                                     torch.tensor([15000.0, 0.0], requires_grad=True),
                                     torch.tensor([0.0, 15000.0], requires_grad=True),
                                     torch.tensor([0.0, 0.], requires_grad=True))

    task = wf_phase2.add_(pf3)

    virtual_fields = task.send(None)
    concrete_fields = []
    if virtual_fields is not None:
        field_builder = ti.FieldsBuilder()
        for vf in virtual_fields:
            concrete_fields.append(vf.concretize(field_builder))
        snode1 = field_builder.finalize()
        for cf, vf in zip(concrete_fields, virtual_fields):
            if vf.requires_grad:
                cf.grad.fill(0)

    virtual_fields = task.send(concrete_fields)
    concrete_fields = []
    if virtual_fields is not None:
        field_builder = ti.FieldsBuilder()
        for vf in virtual_fields:
            concrete_fields.append(vf.concretize(field_builder))
        snode2 = field_builder.finalize()
        for cf, vf in zip(concrete_fields, virtual_fields):
            if vf.requires_grad:
                cf.grad.fill(0)

    try:
        task.send(concrete_fields)
    except StopIteration:
        _write_log("finish")
        pass

    _write_log(torch.allclose(wf_phase.values, wf_phase2.values))

    correct_res = torch.load('./self_count_bf.pt')
    ind = torch.allclose(correct_res, wf_phase.values)
    _write_log(f'CORRECT OR NOT: {ind}')
    _write_log(wf_phase.values[0, 0])


def test_case_0():
    ti.init(ti.cpu)
    values = torch.ones((14, 14)).double()
    values = torch.complex(values, values)
    pf = VectorValuedFunction(values, torch.tensor([0.0, 0.1]), torch.tensor([0.1, 0.0]), torch.tensor([0.0, 0.]))

    values = torch.zeros((440, 440)).double()
    values = torch.complex(values, values)
    wf_phase = VectorValuedFunction(values, torch.tensor([-8 / 15., 0.0]), torch.tensor([0.0, 8 / 15.]), torch.zeros(2))

    wf_phase.add_(pf)

    # pf.add_(wf_phase)
    ind = wf_phase.values.real > 0
    print('\n')
    print(wf_phase.values.real[ind])
    ind = ind.nonzero()
    print(ind)


def test_case_1_ti():
    ti.init(ti.cpu)
    values = torch.ones((14, 14)).double()
    values = torch.complex(values, values)
    pf = VectorValuedFunction(values, torch.tensor([0.1, 0.0]), torch.tensor([0.0, -0.1]), torch.tensor([0, -50.]))

    values = torch.zeros((440, 440)).double()
    values = torch.complex(values, values)
    wf_phase = VectorValuedFunction(values, torch.tensor([8. / 15, 0.0]), torch.tensor([0.0, 8. / 15]),
                                    torch.tensor([0, 0.]))
    # wf_phase.add_ti_(pf)

    wf_phase.add_(pf)
    # print('\n')
    # ind = wf_phase.values.real > 0
    # ind = ind.nonzero()
    # print(ind)


def test_case_1():
    values = torch.ones((14, 14)).double()
    values = torch.complex(values, values)
    pf = VectorValuedFunction(values, torch.tensor([0.1, 0.0]), torch.tensor([0.0, -0.1]), torch.tensor([0, -50.]))

    values = torch.zeros((440, 440)).double()
    values = torch.complex(values, values)
    wf_phase = VectorValuedFunction(values, torch.tensor([8. / 15, 0.0]), torch.tensor([0.0, 8. / 15]),
                                    torch.tensor([0, 0.]))
    wf_phase.add_(pf)
    print('\n')
    ind = wf_phase.values.real > 0
    ind = ind.nonzero()
    print(ind)


def test_case_2():
    ti.init(ti.cpu)
    values = torch.ones((14, 14)).double()
    values = torch.complex(values, values)
    pf = VectorValuedFunction(values, torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([0., 0.]))

    values = torch.zeros((440, 440)).double()
    values = torch.complex(values, values)
    wf_phase = VectorValuedFunction(values, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0, 0.]))
    wf_phase.add_(pf)
    # pf.add_(wf_phase)

    print('\n')
    ind = wf_phase.values.real > 0
    ind = ind.nonzero()
    print(ind.shape)


def test_case_3():
    values = torch.ones((14, 14)).double()
    values = torch.complex(values, values)
    pf = VectorValuedFunction(values, torch.tensor([0.0, 0.1]), torch.tensor([0.1, 0.0]), torch.tensor([0., -50.]))

    values = torch.zeros((440, 440)).double()
    values = torch.complex(values, values)
    wf_phase = VectorValuedFunction(values, torch.tensor([-8 / 15., 0.0]), torch.tensor([0.0, 8 / 15.]),
                                    torch.tensor([0, 0.]))

    values = torch.ones((14, 14)).double()
    values = torch.complex(values, values)
    wf = VectorValuedFunction(values, torch.tensor([-8 / 15., 0.0]), torch.tensor([0.0, 8 / 15.]),
                              torch.tensor([0, 0.]))

    # wf_phase.add_(pf)
    # pf.add_(wf)
    #
    # print('\n')
    # ind = wf_phase.values.real > 0
    # ind = ind.nonzero()
    # print(ind.shape)
    # print(wf_phase.values.real[218:222, 124:128])
    #
    # print('\n')
    # ind = pf.values.real > 0
    # ind = ind.nonzero()
    # print(ind.shape)


def test_case_4():
    values = torch.ones((4, 6)).double()
    values = torch.complex(values, values)
    pf = VectorValuedFunction(values, torch.tensor([2.0, 0.0]), torch.tensor([0.0, 2.0]), torch.tensor([0, 0.]))

    values = torch.zeros((6, 4)).double()
    values = torch.complex(values, values)
    wf_phase = VectorValuedFunction(values, torch.tensor([1.5, 0.0]), torch.tensor([0.0, 1.5]), torch.zeros(2))
    wf_phase.add_(pf)
    # pf.add_(wf_phase)
    print('\n')
    ind = wf_phase.values.real > 0
    print(wf_phase.values.real[ind])
    ind = ind.nonzero()
    print(ind.shape)
    print(ind)


def test_case_5():
    values = torch.ones((5, 4)).double()
    values = torch.complex(values, values)
    pf = VectorValuedFunction(values, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0, 0.]))

    values = torch.zeros((4, 5)).double()
    values = torch.complex(values, values)
    wf_phase = VectorValuedFunction(values, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.zeros(2))
    # wf_phase.add_(pf)
    pf.add_(wf_phase)
    print('\n')
    ind = pf.values.real > 0
    print(pf.values.real[ind])
    ind = ind.nonzero()
    print(ind.shape)
    print(ind)


def test_performance():
    ti.init(ti.cpu)
    values = torch.ones((440, 440)).double()
    values = torch.complex(values, values)
    pf = VectorValuedFunction(values, torch.tensor([16000, 0.0]), torch.tensor([0.0, 16000]), torch.tensor([0, 0.]))

    values = torch.zeros((440, 440)).double()
    values = torch.complex(values, values)
    wf_phase = VectorValuedFunction(values, torch.tensor([16000, 0.0]), torch.tensor([0.0, 16000]), torch.zeros(2))

    start = time.time()
    task = wf_phase.add_(pf)

    virtual_fields = task.send(None)
    concrete_fields = []
    if virtual_fields is not None:
        field_builder = ti.FieldsBuilder()
        for vf in virtual_fields:
            concrete_fields.append(vf.concretize(field_builder))
        snode1 = field_builder.finalize()
        for cf, vf in zip(concrete_fields, virtual_fields):
            if vf.requires_grad:
                cf.grad.fill(0)

    virtual_fields = task.send(concrete_fields)
    concrete_fields = []
    if virtual_fields is not None:
        field_builder = ti.FieldsBuilder()
        for vf in virtual_fields:
            concrete_fields.append(vf.concretize(field_builder))
        snode2 = field_builder.finalize()
        for cf, vf in zip(concrete_fields, virtual_fields):
            if vf.requires_grad:
                cf.grad.fill(0)

    try:
        task.send(concrete_fields)
    except StopIteration:
        _write_log("finish")
        pass
    end = time.time()
    print(f"Time {end - start}")
    print('\n')
