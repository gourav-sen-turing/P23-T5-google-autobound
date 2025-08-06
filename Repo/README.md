# AutoBound: Automatically Bounding Functions

![Continuous integration](https://github.com/google/autobound/actions/workflows/ci-build.yaml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/autobound)

AutoBound is a generalization of automatic differentiation.  In addition to
computing a Taylor polynomial approximation of a function, it computes upper
and lower bounds that are guaranteed to hold over a user-specified
_trust region_.

As an example, here are the quadratic upper and lower bounds AutoBound computes
for the function `f(x) = 1.5*exp(3*x) - 25*(x**2)`, centered at `0.5`, and
valid over the trust region `[0, 1]`.

<div align="center">
<img src="http://raw.githubusercontent.com/google/autobound/main/autobound/example_bounds.png" alt="Example quadratic upper and lower bounds"></img>
</div>

The code to compute the bounds shown in this plot looks like this (see [quickstart](https://colab.research.google.com/github/google/autobound/blob/main/autobound/notebooks/quickstart.ipynb)):

```python
import autobound.jax as ab
import jax.numpy as jnp

f = lambda x: 1.5*jnp.exp(3*x) - 25*x**2
x0 = .5
trust_region = (0, 1)
# Compute quadratic upper and lower bounds on f.
bounds = ab.taylor_bounds(f, max_degree=2)(x0, trust_region)
# bounds.upper(1) == 5.1283045 == f(1)
# bounds.lower(0) == 1.5 == f(0)
# bounds.coefficients == (0.47253323, -4.8324013, (-5.5549355, 28.287888))
```

These bounds can be used for:

*   [Computing learning rates that are guaranteed to reduce a loss function](https://colab.research.google.com/github/google/autobound/blob/main/autobound/notebooks/safe_learning_rates.ipynb)
*   [Upper and lower bounding integrals](https://colab.research.google.com/github/google/autobound/blob/main/autobound/notebooks/bounding_integrals.ipynb)
*   Proving optimality guarantees in global optimization

and more!

Under the hood, AutoBound computes these bounds using an interval arithmetic
variant of Taylor-mode automatic differentiation.  Accordingly, the memory
requirements are linear in the input dimension, and the method is only
practical for functions with low-dimensional inputs.  A reverse-mode algorithm
that efficiently handles high-dimensional inputs is under development.

A detailed description of the AutoBound algorithm can be found in
[this paper](https://arxiv.org/abs/2212.11429).

## Installation

Assuming you have [installed pip](https://pip.pypa.io/en/stable/installation/), you can install this package directly from GitHub with

```bash
pip install git+https://github.com/google/autobound.git
```

or from PyPI with

```bash
pip install autobound
```

You may need to [upgrade pip](https://pip.pypa.io/en/stable/installation/#upgrading-pip) before running these commands.

## Testing

To run unit tests, first install the packages the unit tests depend on with

```bash
pip install autobound[dev]
```

As above, you may need to [install](https://pip.pypa.io/en/stable/installation/) or [upgrade](https://pip.pypa.io/en/stable/installation/#upgrading-pip) `pip` before running this command.

Then, download the source code and run the tests using

```bash
git clone https://github.com/google/autobound.git
python3 -m pytest autobound
```

or

```bash
pip install -e git+https://github.com/google/autobound.git#egg=autobound
python3 -m pytest src/autobound
```

## Limitations

The current code has a few limitations:

*   Only JAX-traceable functions can be automatically bounded.
*   Many JAX library functions are not yet supported.  What _is_
    supported is bounding the squared error loss of a multi-layer perceptron or convolutional neural network that uses the `jax.nn.sigmoid`, `jax.nn.softplus`, or `jax.nn.swish` activation functions.
*   To compute accurate bounds for deeper neural networks, you may need to use
    `float64` rather than `float32`.

## Citing AutoBound

To cite this repository:

```
@article{autobound2022,
  title={Automatically Bounding the Taylor Remainder Series: Tighter Bounds and New Applications},
  author={Streeter, Matthew and Dillon, Joshua V},
  journal={arXiv preprint arXiv:2212.11429},
  url = {http://github.com/google/autobound},
  year={2022}
}
```

*This is not an officially supported Google product.*

# Turn 1: Model B response

## Implementation Plan

Based on my analysis, I need to:

1. **Add the `_pow_kth_deriv_sign()` helper function** that:
- Takes parameters: `x_sign` (sign of x: 1 or -1), `p` (exponent), `k` (derivative order)
- Returns the sign of the k-th derivative of x^p evaluated at a point with sign x_sign
- The k-th derivative of x^p is: p * (p-1) * ... * (p-k+1) * x^(p-k)
- Need to handle:
- When k > p for integer p (derivative is 0)
- Sign of the coefficient product p * (p-1) * ... * (p-k+1)
- Sign of x^(p-k) based on x_sign and whether (p-k) is even/odd

2. **Replace manual factorial computation with `math.factorial()`**:
- Current code has a loop computing factorials manually
- Should use `math.factorial()` instead

3. **Fix the logic for determining increasing/decreasing derivatives**:
- Currently uses `c_pos` and `c_neg` variables to track signs
- Should use the new `_pow_kth_deriv_sign()` function to determine if derivatives are increasing or decreasing

4. **Improve handling of negative numbers with non-integer powers**:
- Current code has some issues with this case
- Need to ensure proper handling when exponent is not an integer

# Turn 3: Model B response

## The response given by model B was exact same as previous, here's the summary

# Summary of Changes in primitive_enclosures.py

Based on my analysis of the code and tests, the following changes have been successfully implemented:

## 1. Added `_pow_kth_deriv_sign()` Helper Function

A new function was added (lines 108-151) that computes the sign of the kth derivative of x^p:

```python
def _pow_kth_deriv_sign(x_sign, p, k):
  """Returns the sign of the kth derivative of x^p for a given sign of x."""
```

Key features:

- Correctly handles the special case where derivatives become 0 for non-negative integer exponents when k > p
- Calculates the sign of the coefficient p*(p-1)*...*(p-k+1)
- Determines the sign of x^(p-k) based on whether p-k is odd or even
- Returns the combined sign (-1, 0, or 1)

## 2. Updated `pow_enclosure()` Function

### a) Factorial Calculation (line 169)

- Changed from manual loop calculation to using `math.factorial(i)`
- This simplifies the code and makes it more readable

### b) Derivative Monotonicity Logic (lines 180-201)

- Uses `_pow_kth_deriv_sign()` to determine whether the degree-th derivative is increasing or decreasing
- Separate logic for positive x (using `deriv_sign_pos`) and negative x (using `deriv_sign_neg`)
- Special handling when the derivative sign is 0 (constant derivative)

### c) Better Handling of Edge Cases

- Maintains proper handling of non-integer exponents with negative bases
- Clear separation between integer and non-integer exponent cases

## Results

All 64 tests are now passing, including:

- 20 `test_pow_kth_deriv_sign` tests that verify the helper function
- `test_enclosure_generator_cbrt` for cube root function
- All other pow_enclosure related tests

The implementation successfully addresses all the requirements mentioned in the PR description while maintaining backward compatibility.
