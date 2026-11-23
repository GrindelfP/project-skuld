"""Cross-check the derivatives used by both precision paths."""
import mpmath as mp
import torch

from heat_pinn import Config, DTYPE, HeatField, sample_sites
from heat_pinn_anchored import AnchoredConfig, AnchoredField
from heat_pinn_precision import CompactField, points_as_mp


def check_double(model, dimension):
    x = torch.rand(10, dimension, dtype=DTYPE, requires_grad=True)
    t = torch.rand(10, dtype=DTYPE, requires_grad=True) * .2
    value = model(x, t)
    dx, dt = torch.autograd.grad(value.sum(), (x, t), create_graph=True)
    laplacian = sum(torch.autograd.grad(dx[:, k].sum(), x, retain_graph=True)[0][:, k]
                    for k in range(dimension))
    channel, gradient, lap = model(x, t, True)
    for calculated, expected in ((channel, value), (gradient[:, :-1], dx),
                                 (gradient[:, -1], dt), (lap, laplacian)):
        torch.testing.assert_close(calculated, expected, atol=1e-11, rtol=1e-11)
    expected = {'value': value[0].item(), 'pde': (dt[0] - laplacian[0]).item(),
                'flux': dx[0, dimension - 1].item()}
    return points_as_mp(x.detach())[0], mp.mpf(t[0].item()), expected


def check_multiprecision(model, point, moment, expected):
    field = CompactField(model)
    for kind, expected_value in expected.items():
        axis = model.dimension - 1
        value, jac = field.evaluate(point, moment, kind, axis, True)
        assert abs(value - mp.mpf(expected_value)) < mp.mpf('1e-12')
        original = field.weights
        for index in range(len(original)):
            weights = original.copy()
            delta = mp.mpf('1e-28')
            weights[index] += delta
            field.weights = weights
            plus = field.evaluate(point, moment, kind, axis)
            weights[index] -= 2 * delta
            minus = field.evaluate(point, moment, kind, axis)
            field.weights = original
            assert abs((plus - minus) / (2 * delta) - jac[index]) < mp.mpf('1e-30')


def check_anchored(dimension, sites):
    for reflected in (False, True):
        model = AnchoredField(dimension, AnchoredConfig(width=6, initial_width=7,
                                                       reflected=reflected, initial_feature=True))
        check_double(model, dimension)
        expected = model.initial(2 * sites.initial - 1)
        torch.testing.assert_close(model(sites.initial, torch.zeros_like(sites.times)), expected)
        if reflected:
            _, gradient, _ = model(sites.faces, sites.face_times, True)
            normal = gradient.gather(1, sites.axes[:, None])
            assert normal.count_nonzero() == 0


def main():
    torch.manual_seed(41)
    with mp.workdps(65):
        for dimension in (1, 3, 6):
            sites = sample_sites(dimension, 512, 42, balanced=True)
            assert set(sites.faces.gather(1, sites.axes[:, None]).flatten().tolist()) == {0., 1.}
            assert 0 < sites.times.min() < .01 < 5 < sites.times.max() < 6
            for compact in (False, True):
                model = HeatField(dimension, Config(width=6, compact=compact, rates=(1.,)))
                point, moment, expected = check_double(model, dimension)
                if compact:
                    check_multiprecision(model, point, moment, expected)
            check_anchored(dimension, sites)
    print('Heat PINN: sampling, derivatives, anchoring, and reflected flux checks passed.')


if __name__ == '__main__':
    main()
