# Copyright 2023 The autobound Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A library of Taylor enclosures for various primitive functions.

For now this only supports elementwise functions, but in the future it will
support other multivariate functions.
"""

import functools
import math
from typing import Callable, Sequence

from autobound import elementwise_functions
# pylint: disable=g-multiple-import
from autobound.types import (
    Interval, NumpyLike, ElementwiseTaylorEnclosure, NDArray, NDArrayLike)


def get_elementwise_taylor_enclosure(
    function_id: elementwise_functions.FunctionId,
    x0: NDArray,
    trust_region: Interval,
    degree: int,
    np_like: NumpyLike) -> ElementwiseTaylorEnclosure:
  """Returns ElementwiseTaylorEnclosure for function with given ID.

  Args:
    function_id: an `elementwise_functions.FunctionId`
    x0: reference point
    trust_region: trust region over which enclosure is valid
    degree: the degree of the returned `ElementwiseTaylorEnclosure`
    np_like: a `NumpyLike` backend

  Returns:
    an `ElementwiseTaylorEnclosure` for the elementwise function specified by
    `function_id`.
  """
  f = elementwise_functions.get_function(function_id, np_like)
  deriv_id = function_id.derivative_id(degree)
  deriv_data = elementwise_functions.get_function_data(deriv_id)
  taylor_coefficients = functools.partial(
      elementwise_functions.get_taylor_polynomial_coefficients,
      function_id, x0=x0, np_like=np_like)
  if (deriv_data.monotonically_increasing or
      deriv_data.monotonically_decreasing):
    return sharp_enclosure_monotonic_derivative(
        x0, degree, trust_region, f, taylor_coefficients(degree),
        deriv_data.monotonically_increasing, np_like)
  elif degree == 2 and deriv_data.even_symmetric:
    return sharp_quadratic_enclosure_even_symmetric_hessian(
        x0, trust_region, f, taylor_coefficients(degree), np_like)
  else:
    # For indices where the derivative is monotonically decreasing or
    # monotonically increasing over the trust region, we return the sharp
    # enclosure.  For other indices, we fall back to using the enclosure
    # based on the range of the derivative.
    coeffs = taylor_coefficients(degree)
    enclosure_if_decreasing, enclosure_if_increasing = [
        sharp_enclosure_monotonic_derivative(
            x0, degree, trust_region, f, coeffs, increasing, np_like)
        for increasing in [False, True]
    ]
    decreasing, increasing = deriv_data.monotone_over(trust_region, np_like)
    deriv_range = elementwise_functions.get_range(deriv_id, trust_region,
                                                  np_like)
    fallback = bounded_derivative_enclosure(degree, coeffs[:-1], deriv_range)
    def endpoint(i: int):
      return np_like.where(
          decreasing,
          enclosure_if_decreasing[-1][i],
          np_like.where(
              increasing,
              enclosure_if_increasing[-1][i],
              fallback[-1][i]
          )
      )
    final_interval = (endpoint(0), endpoint(1))
    return ElementwiseTaylorEnclosure(
        tuple(coeffs[:degree]) + (final_interval,))


abs_enclosure = functools.partial(get_elementwise_taylor_enclosure,
                                  elementwise_functions.ABS)
exp_enclosure = functools.partial(get_elementwise_taylor_enclosure,
                                  elementwise_functions.EXP)
log_enclosure = functools.partial(get_elementwise_taylor_enclosure,
                                  elementwise_functions.LOG)
sigmoid_enclosure = functools.partial(get_elementwise_taylor_enclosure,
                                      elementwise_functions.SIGMOID)
softplus_enclosure = functools.partial(get_elementwise_taylor_enclosure,
                                       elementwise_functions.SOFTPLUS)
swish_enclosure = functools.partial(get_elementwise_taylor_enclosure,
                                    elementwise_functions.SWISH)


def _pow_kth_deriv_sign(x_sign, p, k):
  """Returns the sign of the kth derivative of x^p for a given sign of x.

  Args:
    x_sign: Sign of x (1 or -1)
    p: Exponent
    k: Derivative order

  Returns:
    Sign of the kth derivative: -1, 0, or 1
  """
  # Special case: if p is a non-negative integer and k > p, the derivative is 0
  if isinstance(p, int) and p >= 0 and k > p:
    return 0

  # Calculate the sign of the coefficient: p * (p-1) * ... * (p-k+1)
  coeff_sign = 1
  for i in range(k):
    factor = p - i
    if factor == 0:
      return 0  # Derivative is zero
    elif factor < 0:
      coeff_sign *= -1

  # Now we need to determine the sign of x^(p-k)
  # If p-k == 0, then x^(p-k) = 1, so sign is 1
  if p - k == 0:
    x_power_sign = 1
  else:
    # For x^(p-k), the sign depends on:
    # - If x is positive (x_sign = 1), then x^(p-k) > 0
    # - If x is negative (x_sign = -1), then:
    #   - If (p-k) is even (or non-integer), x^(p-k) > 0
    #   - If (p-k) is odd integer, x^(p-k) has same sign as x
    if x_sign == 1:
      x_power_sign = 1
    else:  # x_sign == -1
      # Check if (p-k) is an odd integer
      if isinstance(p - k, int) and (p - k) % 2 == 1:
        x_power_sign = -1
      else:
        x_power_sign = 1

  return coeff_sign * x_power_sign


# TODO(mstreeter): we could implement pow_enclosure in terms of
# get_elementwise_taylor_enclosure if we allowed FunctionIds to have parameters
# (in this case, the exponent).
def pow_enclosure(exponent: float,
                  x0: NDArray,
                  trust_region: Interval,
                  degree: int,
                  np_like: NumpyLike) -> ElementwiseTaylorEnclosure:
  """Returns an ElementwiseTaylorEnclosure for x**exponent in terms of x-x0."""
  taylor_coefficients_at_x0 = []
  c = 1.
  for i in range(degree + 1):
    # Note: the next line can sometimes generate bogus RuntimeWarnings when
    # using Numpy.  This seems to be a bug in Numpy, as even doing
    # np.array(2.)**-1 generates the same RuntimeWarning.
    taylor_coefficients_at_x0.append(c * x0**(exponent - i) / math.factorial(i))
    c *= exponent - i

  enc_decreasing, enc_increasing = [
      sharp_enclosure_monotonic_derivative(
          x0, degree, trust_region, lambda x: x**exponent,
          taylor_coefficients_at_x0, increasing, np_like
      )
      for increasing in [False, True]
  ]

  # Determine whether the degree-th derivative is increasing or decreasing
  # for positive x by checking the sign of the (degree+1)-th derivative
  deriv_sign_pos = _pow_kth_deriv_sign(1, exponent, degree + 1)
  if deriv_sign_pos == 0:
    # When (degree+1)-th derivative is 0, the degree-th derivative is constant
    # In this case, both enc_increasing and enc_decreasing should give the same result
    # but we use enc_decreasing by convention
    enc_pos = enc_decreasing
  else:
    enc_pos = (enc_increasing if deriv_sign_pos > 0 else enc_decreasing)

  if int(exponent) != exponent:
    # For non-integer exponents, x^p is undefined for negative x
    enc_neg = None
  else:
    # For integer exponents, determine monotonicity for negative x
    deriv_sign_neg = _pow_kth_deriv_sign(-1, exponent, degree + 1)
    if deriv_sign_neg == 0:
      # When (degree+1)-th derivative is 0, the degree-th derivative is constant
      enc_neg = enc_decreasing
    else:
      enc_neg = (enc_increasing if deriv_sign_neg > 0 else enc_decreasing)

  def interval_endpoint(i):
    """Returns left (i == 0) or right (i == 1) endpoint of interval."""
    a, b = trust_region
    # For each index i, the interval between a[i] and b[i] either contains zero,
    # or contains only positive values, or contains only negative values.
    # Compute the endpoints for all three cases, and combine the results using
    # np_like.where().
    endpoint_if_always_positive = enc_pos[-1][i]
    if int(exponent) != exponent:
      # If exponent is not an integer, then z**exponent is undefined for z < 0.
      endpoint_if_always_negative = 0
      endpoint_if_possibly_zero = 0
    elif exponent < 0:
      endpoint_if_always_negative = enc_neg[-1][i]
      endpoint_if_possibly_zero = -np_like.inf if i == 0 else np_like.inf
    else:
      endpoint_if_always_negative = enc_neg[-1][i]
      endpoint_if_possibly_zero = functools.reduce(
          np_like.minimum if i == 0 else np_like.maximum,
          [endpoint_if_always_positive, endpoint_if_always_negative]
      )
    return np_like.where(
        a >= 0,
        endpoint_if_always_positive,
        np_like.where(
            b <= 0,
            endpoint_if_always_negative,
            endpoint_if_possibly_zero
        )
    )

  interval_coefficient = tuple(interval_endpoint(i) for i in [0, 1])
  return ElementwiseTaylorEnclosure(
      enc_decreasing[:-1] + (interval_coefficient,))


def bounded_derivative_enclosure(
    degree: int,
    taylor_coefficients_at_x0: Sequence[NDArray],
    derivative_bound: tuple[NDArray, NDArray]
) -> ElementwiseTaylorEnclosure:
  if len(taylor_coefficients_at_x0) != degree:
    raise ValueError()
  degree_factorial = math.factorial(degree)
  final_interval = (derivative_bound[0] / degree_factorial,
                    derivative_bound[1] / degree_factorial)
  return ElementwiseTaylorEnclosure(
      tuple(taylor_coefficients_at_x0[:degree]) + (final_interval,)
  )


def sharp_enclosure_monotonic_derivative(
    x0: NDArray,
    degree: int,
    trust_region: Interval,
    sigma: Callable[[NDArray], NDArray],
    taylor_coefficients_at_x0: Sequence[NDArray],
    increasing: bool,
    np_like: NumpyLike
) -> ElementwiseTaylorEnclosure:
  """Returns sharp degree-k enclosure assuming monotone k-th derivative.

  Args:
    x0: the center point for the Taylor enclosure
    degree: the degree of the enclosure to return
    trust_region: the trust region over which to compute an enclosure
    sigma: the function for which to compute a sharp polynomial enclosure
    taylor_coefficients_at_x0: the first (degree+1) coefficients of the
      Taylor series expansion of sigma at x0.
    increasing: whether the (degree)th derivative of sigma is increasing
      or decreasing
    np_like: a NumpyLike backend

  Returns:
    a sharp ElementwiseTaylorEnclosure for sigma
  """
  if degree < 0:
    raise ValueError(degree)
  ratio = functools.partial(taylor_remainder_ratio,
                            x0, degree, sigma,
                            taylor_coefficients_at_x0,
                            np_like=np_like)
  a, b = trust_region
  if increasing:
    final_interval = (ratio(a), ratio(b))
  else:
    final_interval = (ratio(b), ratio(a))
  return ElementwiseTaylorEnclosure(
      tuple(taylor_coefficients_at_x0[:degree]) + (final_interval,)
  )


def sharp_quadratic_enclosure_even_symmetric_hessian(
    x0: NDArray,
    trust_region: Interval,
    sigma: Callable[[NDArray], NDArray],
    taylor_coefficients_at_x0: Sequence[NDArray],
    np_like: NumpyLike
) -> ElementwiseTaylorEnclosure:
  """Returns sharp quadratic enclosure for function with even-symmetric Hessian.

  It's assumed that the Hessian is decreasing at z >= 0.

  Args:
    x0: the center point for the Taylor enclosure
    trust_region: the trust region over which to compute an enclosure
    sigma: an elementwise function for which to compute a Taylor enclosure
    taylor_coefficients_at_x0: the first two coefficients of the
      Taylor series expansion of sigma at x0.
    np_like: a Numpy-like back end.
  """
  ratio = functools.partial(taylor_remainder_ratio,
                            x0, 2, sigma,
                            taylor_coefficients_at_x0,
                            np_like=np_like)

  a, b = trust_region
  max_ratio = ratio(np_like.clip(-x0, a, b))
  min_ratio = np_like.minimum(ratio(a), ratio(b))
  final_interval = (min_ratio, max_ratio)
  return ElementwiseTaylorEnclosure(
      tuple(taylor_coefficients_at_x0[:2]) + (final_interval,)
  )


def taylor_remainder_ratio(
    x0: NDArray,
    degree: int,
    sigma: Callable[[NDArray], NDArray],
    taylor_coefficients_at_x0: Sequence[NDArray],
    x: NDArray,
    np_like: NumpyLike):
  """Returns R_{degree - 1}(x; sigma, x0) / (x - x0)**degree."""
  if len(taylor_coefficients_at_x0) != degree + 1:
    raise ValueError(degree, taylor_coefficients_at_x0)
  # Let r_k denote the degree k Taylor series remainder.
  #
  # Letting k = degree, we want to return r_{k-1} / (x-x0)**k, but in a way that
  # is numerically stable when x-x0 is small (and that is well-defined when
  # x=x0).
  #
  # We do so using:
  #     r_{k-1} / (x-x0)**k = (r_k + c_k*(x-x0)**k) / (x-x0)**k
  #                         = c_k + r_k / (x-x0)**k.
  r_k = sigma(x) - sum(
      c * (x - x0)**i for i, c in enumerate(taylor_coefficients_at_x0))
  denom = (x-x0)**degree
  return (
      taylor_coefficients_at_x0[degree] +
      # Return r_k * 1 / denom, capping the magnitude of 1 / denom at 1e12.
      # TODO(mstreeter): this results in an enclosure that's  not strictly valid
      # when denom is very small.
      r_k * np_like.sign(denom) / (np_like.maximum(1e-12, np_like.abs(denom)))
  )
