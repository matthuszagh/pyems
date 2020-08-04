from multiprocessing import Pool
import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.special import polygamma
from pyems.physical_constant import C0, MUE0, Z0
from pyems.utilities import print_table


def wheeler_z0(w: float, t: float, er: float, h: float) -> float:
    """
    Calculate the microstrip characteristic impedance for a given
    width using Wheeler's equation.  Wheeler's equation can be found
    at:

    https://en.wikipedia.org/wiki/Microstrip#Characteristic_impedance

    :param w: microstrip trace width (m)
    :param t: trace thickness (m)
    :param er: substrate relative permittivity
    :param h: substrate height (thickness) (m)

    :returns: characteristic impedance
    """
    weff = w + (
        (t * ((1 + (1 / er)) / (2 * np.pi)))
        * np.log(
            (4 * np.e)
            / (
                np.sqrt(
                    ((t / h) ** 2)
                    + (((1 / np.pi) * ((1 / ((w / t) + (11 / 10))))) ** 2)
                )
            )
        )
    )
    tmp1 = 4 * h / weff
    tmp2 = (14 + (8 / er)) / 11
    zm = (Z0 / (2 * np.pi * np.sqrt(2 * (1 + er)))) * np.log(
        1
        + (
            tmp1
            * (
                (tmp2 * tmp1)
                + (
                    np.sqrt(
                        (
                            (tmp2 * tmp1) ** 2
                            + ((np.pi ** 2) * ((1 + (1 / er)) / 2))
                        )
                    )
                )
            )
        )
    )
    return zm


def wheeler_z0_width(
    z0: float,
    t: float,
    er: float,
    h: float,
    tol: float = 0.01,
    guess: float = 0.3,
) -> float:
    """
    Calculate the microstrip width for a given characteristic
    impedance using Wheeler's formula.

    :param z0: characteristic impedance (ohm)
    :param t: trace thickness (m)
    :param er: substrate relative permittivity
    :param h: substrate height (thickness) (m)
    :param tol: acceptable impedance tolerance (ohm)
    :param guess: an initial guess for the width (m).  This can
        improve convergence time when the approximate width is known.

    :returns: trace width (m)
    """
    width = guess
    zm = wheeler_z0(w=width, t=t, er=er, h=h)
    wlow = width / 10
    zlow = wheeler_z0(w=wlow, t=t, er=er, h=h)
    # inverse relation between width and z0
    while zlow < z0:
        wlow /= 10
        zlow = wheeler_z0(w=wlow, t=t, er=er, h=h)

    whigh = width * 10
    zhigh = wheeler_z0(w=whigh, t=t, er=er, h=h)
    while zhigh > z0:
        whigh *= 10
        zhigh = wheeler_z0(w=whigh, t=t, er=er, h=h)

    while np.absolute(zm - z0) > tol:
        if zm > z0:
            m = (zhigh - zm) / (whigh - width)
            wlow = width
            zlow = zm
        else:
            m = (zm - zlow) / (width - wlow)
            whigh = width
            zhigh = zm

        # use linear interpolation to update guess
        width = width + ((z0 - zm) / m)
        zm = wheeler_z0(w=width, t=t, er=er, h=h)

    return width


def pozar_z0(
    trace_width: float, substrate_height: float, substrate_dielectric: float
) -> float:
    """
    Calculate the characteristic impedance of a microstrip
    transmission line using the approximation in Pozar 4e p.148.
    """
    eeff = microstrip_effective_dielectric(
        substrate_dielectric, substrate_height, trace_width
    )
    wh_ratio = trace_width / substrate_height
    if wh_ratio <= 1:
        return (60 / np.sqrt(eeff)) * np.log((8 / wh_ratio) + (wh_ratio / 4))
    else:
        return (
            120
            * np.pi
            / (
                np.sqrt(eeff)
                * (wh_ratio + 1.393 + 0.667 * np.log(wh_ratio + 1.444))
            )
        )


def pozar_z0_width(
    z0: float, substrate_height: float, substrate_dielectric: float
) -> float:
    """
    Calculate the microstrip trace width for a desired characteristic
    impedance using the approximation in Pozar 4e p.148.
    """
    a = ((z0 / 60) * np.sqrt((substrate_dielectric + 1) / 2)) + (
        ((substrate_dielectric - 1) / (substrate_dielectric + 1))
        * (0.23 + (0.11 / substrate_dielectric))
    )
    b = 377 * np.pi / (2 * z0 * np.sqrt(substrate_dielectric))
    ratio1 = 8 * np.exp(a) / (np.exp(2 * a) - 2)
    ratio2 = (2 / np.pi) * (
        b
        - 1
        - np.log(2 * b - 1)
        + ((substrate_dielectric - 1) / (2 * substrate_dielectric))
        * (np.log(b - 1) + 0.39 - 0.61 / substrate_dielectric)
    )
    if ratio1 < 2 and ratio2 < 2:
        return ratio1 * substrate_height
    elif ratio1 > 2 and ratio2 > 2:
        return ratio2 * substrate_height
    else:
        raise RuntimeError("Conflicting ratios.")


def miter(trace_width: float, substrate_height: float) -> float:
    """
    Compute the optimal miter length using the Douville and James
    equation.
    """
    if trace_width / substrate_height < 0.25:
        raise ValueError(
            "Ratio of trace width to height must be at least 0.25."
        )
    d = trace_width * np.sqrt(2)
    x = d * (
        0.52 + (0.65 * np.exp(-(27 / 20) * trace_width / substrate_height))
    )
    return x


def coax_core_diameter(
    outer_diameter: float, permittivity: float, impedance: float = 50
) -> float:
    """
    Approximate coaxial cable core diameter for a given target
    characteristic impedance and outer diameter.

    :returns: Inner core diameter.  The units will match those of the
              provided outer diameter.
    """
    return outer_diameter / np.power(
        10, impedance * np.sqrt(permittivity) / 138
    )


def microstrip_effective_dielectric(
    substrate_dielectric: float, substrate_height: float, trace_width: float
) -> float:
    """
    Compute the effective dielectric for a microstrip trace.  This
    represents the dielectric constant for a homogenous medium that
    replaces the air and substrate.  See Pozar 4e p.148 and Wadell
    p.94 for details.

    :param substrate_dielectric: Dielectric constant of the PCB
        substrate.
    :param substrate_height: Distance between the backing ground plane
        and microstrip trace.
    :param trace_width: Microstrip trace width.  Any units can be used
        for substrate_height and trace_width as long as they are
        consistent.
    """
    wh_ratio = trace_width / substrate_height
    common_factor = 1 / np.sqrt(1 + (12 / wh_ratio))

    if wh_ratio < 1:
        factor = common_factor + (0.04 * ((1 - wh_ratio) ^ 2))
    else:
        factor = common_factor

    return ((substrate_dielectric + 1) / 2) + (
        ((substrate_dielectric - 1) / 2) * factor
    )


def phase_shift_length(
    phase_shift: float, dielectric: float, frequency: float
) -> float:
    """
    Compute the length (in mm) for a signal to undergo a given phase
    shift.  When computing this value for a transmission line not
    surrounded by a homogenous medium (e.g. a microstrip trace), make
    sure to use the effective dielectric.

    :param phase_shift: Phase shift in degrees.
    :param dielectric: Dielectric or effective dielectric constant.
    :param frequency: Signal frequency.
    """
    rad = phase_shift * np.pi / 180
    vac_lambda = 2 * np.pi * frequency / speed_of_light(unit=1e-3)
    return rad / (np.sqrt(dielectric) * vac_lambda)


def skin_depth(
    frequency: float, resistivity: float = 1.68e-8, rel_permeability: float = 1
) -> float:
    """
    Compute the skin depth for a conductor at a given frequency.  The
    default values are for copper.
    """
    return np.sqrt(
        2 * resistivity / (2 * np.pi * frequency * rel_permeability * MUE0)
    )


def speed_of_light(unit: float) -> float:
    """
    """
    return C0 / unit


def wavelength(freq: np.array, unit: float) -> np.array:
    """
    Calculate the wavelength for a given frequency of light.  This
    presently assumes that the light is travelling through a vacuum.
    """
    return speed_of_light(unit) / freq


def wavenumber(freq: np.array, unit: float) -> np.array:
    """
    Calculate the wavenumber for a given frequency of light.  Assumes
    light is travelling through a vacuum.
    """
    return np.array(2 * np.pi / wavelength(freq, unit))


def sweep(func, params, processes: int = 5):
    """
    :param func: Function object taking one item in the `params` list.
    :param params: List of parameter values to sweep over.
    :param processes: Max number of parallel processes.
    """
    pool = Pool(processes=processes)
    ret_vals = list(pool.map(func, params))
    return ret_vals


def optimize_parameter(
    func, start, step, tol, max_steps, display_progress=False
):
    """
    Compute the lowest-cost value of a parameter that still produces
    accurate results.
    """
    res1 = func(start)
    n = len(res1)
    res_matrix = np.zeros((max_steps, n), dtype=np.clongdouble)
    res_matrix[0] = np.array(res1)
    # compute the root mean square differences from the previous results array.
    rms = np.zeros((max_steps - 1,))
    # If the first few RMS values don't show a clear trend, ignore
    # them, since this may have been the result of poor start value
    # choice.
    rms_trend_down = False
    rms_valid_index = 0
    i = 1
    orig_start = start
    start += step

    while i < max_steps:
        res_matrix[i] = np.array(func(start))
        diff = np.subtract(res_matrix[i], res_matrix[i - 1])
        rms[i - 1] = np.sqrt(
            np.sum(np.real(np.multiply(diff, np.conj(diff)))) / n
        )
        if display_progress:
            num_steps = int(np.round((start - orig_start) / step) + 1)
            print_table(
                np.abs(res_matrix[: i + 1]),
                [
                    "{:.4f}".format(val)
                    for val in np.linspace(orig_start, start, num=num_steps)
                ],
                [4 for _ in range(num_steps)],
            )
            print("parameter: {}".format(start))
            print("RMS: {:.10f}".format(rms[i - 1]))

        if not rms_trend_down and i > 1 and rms[i - 1] > rms[i - 2]:
            rms_valid_index = i - 1

        if i - rms_valid_index > 2:
            if (
                not rms_trend_down
                and rms[i - 1] < rms[i - 2]
                and rms[i - 2] < rms[i - 3]
            ):
                rms_trend_down = True

            valid_start = orig_start + (rms_valid_index * step)
            num_valid_steps = int(
                np.round((start - step - valid_start) / step) + 1
            )
            try:
                print(
                    np.linspace(valid_start, start - step, num=num_valid_steps)
                )
                print(rms[rms_valid_index:i])
                fit = curve_fit(
                    rms_fit,
                    np.linspace(
                        valid_start, start - step, num=num_valid_steps
                    ),
                    rms[rms_valid_index:i],
                )
                a = fit[0][0]
                b = fit[0][1]
                error_estimate = rms_remaining_sum(a, b, start + 1)

                if display_progress:
                    print("Error estimate: {:.10f}".format(error_estimate))

                if error_estimate < tol:
                    return start
            except RuntimeError:
                # If curve fitting fails, ignore the oldest RMS value
                # and try again on the next iteration.
                print(
                    "Failed to fit curve to RMS values, ignoring oldest value."
                )
                rms_valid_index += 1

        i += 1
        start += step

    raise RuntimeError(
        "Failed to optimize parameter. Consider increasing "
        "the tolerance or max number of steps."
    )


def rms_fit(x, a, b):
    """
    """
    return np.divide(a, np.power(np.subtract(x, b), 2))


def rms_remaining_sum(a, b, c):
    """
    Computes the sum: sum(a/(x-b)^2, x=c, oo)
    """
    return a * polygamma(1, c - b)


def minimize(func, initial, tol, bounds=None):
    """
    Thin wrapper around scipy.optimize.minimize.

    I've added this so that you can call minimize directly from pyems
    and to make the calling syntax slightly easier (though less
    flexible).

    :param func: Function object that takes 1 or more arguments and
        returns a result to be minimized.
    :param initial: Initial values passed to `func`.  This must be a
        single value, or an array of values if `func` takes more than
        1 argument.
    :param bounds: Constrains parameters to a range of values.  This
        is a tuple of (min, max) for each parameter.  Use None for no
        bounds.  The default for each parameter is no bounds.  A
        single value of None can be used to not place boundaries on
        any parameters.
    :param tol: Result tolerance.

    :returns: Array of function arguments that minimize the function,
              or a single value if `func` only takes 1 argument.

    Example invocation:
    res = minimize(func=func, initial=[1.2], tol=1e-2, bounds=[(0, None)])
    """
    if bounds is None:
        if type(initial) is list:
            num_params = len(initial)
        else:
            num_params = 1
        bounds = []
        for i in num_params:
            bounds.append((None, None))
        bounds = tuple(bounds)

    res = scipy.optimize.minimize(
        func,
        np.array(initial),
        method="L-BFGS-B",
        tol=tol,
        bounds=bounds,
        options={"disp": True},
    )
    if not res.success:
        raise RuntimeError(
            "Minimization failed. See scipy.optimize.minimize "
            "for other options."
        )

    if len(res.x) == 1:
        return res.x[0]
    else:
        return res.x
