import numpy as np
from pyems.physical_constant import C0, MUE0


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
    z0 = 376.730313668
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
    zm = (z0 / (2 * np.pi * np.sqrt(2 * (1 + er)))) * np.log(
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


def miter(trace_width: float, substrate_height: float) -> float:
    """
    Compute the optimal miter length using the Douville and James
    equation.
    """
    if trace_width / substrate_height < 0.25:
        raise ValueError(
            "Ratio of trace width to height must be at least 0.25."
        )
    return (
        trace_width
        * np.sqrt(2)
        * (0.52 + (0.65 * np.exp(-(27 / 20) * trace_width / substrate_height)))
    )


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
    replaces the air and substrate.  See Pozar 4e p.148 for details.

    :param substrate_dielectric: Dielectric constant of the PCB
        substrate.
    :param substrate_height: Distance between the backing ground plane
        and microstrip trace.
    :param trace_width: Microstrip trace width.  Any units can be used
        for substrate_height and trace_width as long as they are
        consistent.
    """
    return ((substrate_dielectric + 1) / 2) + (
        ((substrate_dielectric - 1) / 2)
        * (1 / np.sqrt(1 + (12 * substrate_height / trace_width)))
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
