# mvrsquared v0.1.6
This patch addresses the issue with package documentation highlighted by CRAN and
described [here](https://github.com/r-lib/roxygen2/issues/1491)

It also cites the arXiv paper in the DESCRIPTION by its DOI,
`<doi:10.48550/arXiv.1911.11061>`, rather than as `<arXiv:1911.11061>`, which
CRAN's checks had been flagging with a NOTE.

This patch also makes `calc_rsquared()` substantially faster, with no change to
its arguments or return value.

* **Sparse `y` is no longer read as though it were dense.** The calculation
  visited every cell of `y` and read each one out of the sparse matrix, which
  costs a search per read; on a 70,676 x 20,926 matrix that was roughly three
  billion searches to read four million nonzeros. Both sums of squares split
  into a part over the nonzeros and a part that does not depend on `y`'s values,
  and the second part has a closed form, so only the nonzeros need visiting.
  On that matrix with 100 latent dimensions: **194 s to 1.5 s single threaded,
  and 16.6 s to 0.34 s on twelve threads.**

* **A `yhat` supplied directly no longer builds an identity matrix.** When
  `yhat` was given as a matrix rather than as two matrices to multiply, it was
  turned into a factorization by pairing it with an identity the width of `y` --
  quadratic memory in the number of columns, and a matrix multiplication that
  did nothing. Such a `yhat` now goes to its own routine. At 20,926 columns the
  identity alone was 3.5 GB, so this case previously became unusable on wide
  outcomes well before it became slow.

* **Results are unchanged**, to floating point tolerance. The sums are
  accumulated in a different order, so this is not bit-for-bit identical to
  0.1.5; agreement with a dense reference calculation is to about 1e-14
  relative, and R-squared is unchanged in every digit `print()` will show you.

* **The note about parallelism and precision has been retired.** Results do not
  depend on `threads`: each observation writes its own slot and the totals are
  summed sequentially afterwards. Thread counts from 1 to 12 were verified to
  give bit-identical results, on both the factorized and direct paths.

# mvrsquared v0.1.5
This patch updates C++11 requirement consistent with current CRAN compilers.

# mvrsquared v0.1.4
This patch fixes an error on r-devel-linux-x86_64-debian-clang in CRAN checks.

# mvrsquared v0.1.3
This patch fixes a parallel issue that could've caused incorrect computations.

# mvrsquared v0.1.2
This patches an error thrown when using large data sets.

# mvrsquared v0.1.1
This patches an error being thrown during testing on some Linux operating systems.
The root cause seems to be an imprecise calculation introduced in parallel computing.
See the note under `help(calc_rsquared)`.

# mvrsquared v0.1.0 
This version introduces parallel processing at the C++ level using RcppThread.

To calculate R-squared in parallel, set the `threads` argument to a number 
greater than 1 when calling `calc_rsquared`.

# mvrsquared v0.0.3
This version makes some changes to documentation including the README

# mvrsquared v0.0.2
This version includes

* An arXiv citation to the working paper deriving this method
* Changes to examples requested by CRAN

# mvrsquared v0.0.1
This version is the first release!

