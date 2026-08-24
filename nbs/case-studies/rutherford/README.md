# GEIGER-1911 — planning an experiment to tell two atoms apart

A worked case study that uses `axiom` to **design an experiment before it is run**: where
to point the detector, what shape to cut the aperture, how long to leave it open, and when
to stop. Every number in the five notebooks is computed; the world they share lives in
[`scattering.py`](scattering.py) beside them.

## The question

> *Is the positive charge of an atom concentrated in a small hard centre, or spread
> through the whole atom? Both models fit everything known in 1910. What experiment would
> tell them apart — and specifically, which scattering angles, and what should the slit
> look like?*

An alpha beam on a gold foil. The two atoms predict different angular distributions of the
scattered particles, and the whole of the design work follows from noticing that they are
**one model at two values of one number**: the radius $R$ of the positive charge.

An alpha aimed head-on stops at $D = 2zZe^2/4\pi\varepsilon_0 E$ — 59 fm for radium-C'
alphas on gold — and a trajectory scattered through $\theta$ comes no closer than
$\tfrac{D}{2}(1 + 1/\sin(\theta/2))$. So a charge ball of radius $R$ kills the scattering
past $\sin(\theta_\text{cut}/2) = 1/(2R/D - 1)$, and the two hypotheses are two values of
that one parameter, six decades apart. Turning a debate into a parameter is what gives the
experiment a resolving power, a sensitivity curve and a stopping rule.

## The answer

**Which angles.** Four roles, and no single angle plays two of them:

| | | why |
|---|---|---|
| **1.0°, 2.5°** | anchor | the only place the multiple-scattering core and the tail amplitude can be measured. Below 0.9° the slit cannot be cut narrow enough |
| **5°, 10°, 20°, 45°** | bank | where the information is, *if* the model of the core is believed |
| **90°, 150°** | witness | where the evidence survives that model being attacked. 150° because that is past what a core thirty times too wide could reach |
| **90°, foil out** | background | nothing else measures it, and every wide-angle claim rests on it |

**What the slit should look like.** An **annular slot**, not a hole. A slit covering azimuth
$\phi$ and $\theta \pm \delta$ subtends exactly $\phi \cdot 2\sin\theta\sin\delta$, so the
same counts can be bought in either direction — and only $\delta$ smears, because the
pattern does not depend on azimuth. So hold the radial half-width at the workshop's floor
(0.05°) and take every additional steradian out in azimuth, opening radially only once the
arc has closed into a full annulus:

| station | 1.0° | 2.5° | 5° | 10° | 20° | 45° | 90° | 150° |
|---|---|---|---|---|---|---|---|---|
| radial half-width | 0.05° | 0.05° | 0.05° | 0.05° | 0.05° | 0.20° | 0.91° | 1.82° |
| azimuthal arc | 0.1° | 0.2° | 2° | 16° | 126° | 360° | 360° | 360° |

A round hole of the same area at 90° would have to be ±13° wide and would report a rate
five per cent too high belonging to no angle in particular. The slot reports it to a part
in four thousand. Same counts, fourteen times the angular resolution, free.

**How long.** A hundred and ten of the two hundred hours at 150° — the opposite of what
evidence-per-hour says, and right, because evidence-per-hour is computed inside the model
that is under attack.

**What comes back.** *R < 33 fm*, against a floor of 29.6 fm that no amount of counting can
beat because it is set by the beam energy alone. Rutherford published $3.4\times10^{-14}$ m
in 1911.

## The finding

Under the model as fitted, an hour at five degrees is worth twenty hours at a hundred and
fifty. Allow an opponent to widen the multiple-scattering core — the one free story that
explains a wide-angle count away — and five degrees loses 99.98 per cent of its evidence
while **a hundred and fifty degrees loses none**.

The wide-angle station is not where the information is. It is where the information that
cannot be argued with is, and those are different places.

## The notebooks

| | | reaches into |
|---|---|---|
| [1](01-two-atoms.ipynb) | two atoms, one parameter, and what the beam can never resolve | `core.BASES`, `core.ModelSpec`, `core.Apply`/`Pow`, `design.fisher_information` |
| [2](02-which-angles.ipynb) | which angles, believed and attacked | `design.estimable_combinations`, `design.identifiability_ridge`, `design.expected_posterior_sd`, `design.design_to_identify`, `design.eig_gaussian` |
| [3](03-the-slit.ipynb) | what shape to cut in the brass | `Scattering.forward` integrated over the aperture, `scattering.slit_for` |
| [4](04-how-long.ipynb) | how the hours split, and when to stop | `design.power_from_se`, `design.obrien_fleming`, `design.monitor`, `design.operating_characteristics` |
| [5](05-running-it.ipynb) | running it, in both worlds | `design.profile_likelihood`, `core.Interval`, `core.Verdict`, `display.Card` |

## Notes on the modelling

**Counting is Poisson; the design math is Gaussian.** `Scattering.forward` returns
$2\sqrt{\mu}$, the variance-stabilizing transform of a Poisson count. The bridge is exact,
not approximate: $(\partial 2\sqrt\mu/\partial\psi)^2/1^2 = (\partial\mu/\partial\psi)^2/\mu$,
which is the Poisson information. So `fisher_information(..., noise_sd=1.0)` on this
surface *is* the Poisson information, and notebook 1 checks it against a longhand
calculation.

**One `forward()`.** The mean is a hand-built expression tree — `Apply(sin)`, `Pow`,
`Apply(exp)` — and everything else reads it: the aperture integral in notebook 3, the
Fisher information in notebooks 2 and 4, and the likelihood in notebook 5. `counts()` is
`(forward/2)²`, so even the expected count comes back through the same evaluation rather
than a second copy of the formula.

**A charge nobody can see.** At the hard-centre truth `lam` is a dead column and
`expected_posterior_sd` hands its prior straight back. That is a design result, not a bug:
the experiment returns an upper bound on the size of the positive charge and never a value,
and the design math says so before any apparatus exists.
