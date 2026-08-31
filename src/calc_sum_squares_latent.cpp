#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::depends(RcppThread)]]
#include "RcppThread.h"
#include <cmath>

using namespace Rcpp ;

// Sum of squared errors and total sum of squares, for a latent-variable
// prediction Yhat = X * W.
//
// WHY THIS DOES NOT LOOP OVER EVERY CELL.
//
// The obvious implementation walks all V columns for each of the D observations
// and reads Y(v, d) as it goes. That is O(D * V * K) arithmetic no matter how
// sparse Y is, and --- worse --- every read of Y(v, d) on a sparse matrix is a
// binary search within a column. On a 70,676 x 20,926 document-term matrix that
// is roughly three billion searches to read four million nonzeros.
//
// Both sums split into a part over Y's nonzeros and a part that does not depend
// on Y's values at all, and the second part collapses:
//
//   sum_v Yhat(v,d)^2  =  x_d' (W W') x_d      with G = W W' formed once, K x K
//   sum_v ybar(v)^2                            a constant across observations
//
// So, writing nz(d) for the nonzero columns of observation d:
//
//   SSE_d = ( x_d' G x_d - sum_{nz} Yhat^2 ) + sum_{nz} (Y - Yhat)^2
//   SST_d = ( sum_v ybar^2 - sum_{nz} ybar^2 ) + sum_{nz} (Y - ybar)^2
//
// which is O(nnz_d * K + K^2) per observation and never touches a zero.
//
// THE PARENTHESES ARE DELIBERATE. The algebraically simpler form
// sum(a^2) - 2*sum(a*b) + sum(b^2) is the same in exact arithmetic but subtracts
// large nearly-equal quantities when the fit is good, losing precision exactly
// where the answer matters most. Grouping as above computes the nonzero terms
// directly --- no cancellation --- and uses the expansion only for the all-zero
// remainder, where both quantities are sums of squares and the difference is
// itself nonnegative.
//
// DETERMINISM. Each observation writes its own slot in SSE/SST and the totals
// are summed sequentially afterwards, so the result does not depend on how the
// work was scheduled or on how many threads ran it.

// [[Rcpp::export]]
NumericVector calc_sum_squares_latent(
    const arma::sp_mat& Y,
    const arma::mat& X,
    const arma::mat& W,
    const arma::vec& ybar,
    int threads
) {

  // Transpose once so that an observation's entries are one contiguous column;
  // arma sparse matrices are column-major, and this is what lets the loop below
  // walk nonzeros instead of searching for them.
  const arma::sp_mat Yt = Y.t();

  const arma::uword n_obs = Yt.n_cols;

  NumericVector result(2);
  arma::vec SSE(n_obs, arma::fill::zeros);
  arma::vec SST(n_obs, arma::fill::zeros);

  // G = W W', so that sum_v Yhat(v,d)^2 is the quadratic form x_d' G x_d.
  const arma::mat G = W * W.t();

  // sum_v ybar(v)^2, the same for every observation.
  const double sum_ybar_sq = arma::dot(ybar, ybar);

  RcppThread::parallelFor(
    0,
    n_obs,
    [&Yt, &X, &W, &ybar, &G, sum_ybar_sq, &SSE, &SST] (unsigned int d) {
      RcppThread::checkUserInterrupt();

      const arma::rowvec x_d = X.row(d);

      // The all-column terms, before removing the nonzeros' share below.
      double sse = arma::as_scalar(x_d * G * x_d.t());
      double sst = sum_ybar_sq;

      for (arma::sp_mat::const_col_iterator it = Yt.begin_col(d);
           it != Yt.end_col(d); ++it) {

        const arma::uword v = it.row();
        const double y = (*it);

        // Yhat for this one entry: the dot product of x_d with W's v-th column.
        const double yhat = arma::dot(x_d, W.col(v));

        // Swap this column's contribution out of the all-column term and put
        // the exact residual in its place.
        sse += -(yhat * yhat) + (y - yhat) * (y - yhat);
        sst += -(ybar[v] * ybar[v]) + (y - ybar[v]) * (y - ybar[v]);
      }

      SSE(d) = sse;
      SST(d) = sst;
    },
    threads);

  result[ 0 ] = sum(SSE);
  result[ 1 ] = sum(SST);

  return result;

}


// Sum of squared errors and total sum of squares when Yhat is supplied directly
// rather than as a factorization.
//
// This exists because the caller used to fake a factorization for a dense Yhat
// by passing X = Yhat and W = diag(V), which costs O(V^2) memory and turns an
// O(D * V) computation into O(D * V^2). At V = 20,926 that identity matrix alone
// is 3.5 GB.
//
// Same decomposition as above, minus the Gram matrix: the all-column term for
// SSE is just the sum of squares of Yhat's row, which is already dense and
// therefore free to take directly.

// [[Rcpp::export]]
NumericVector calc_sum_squares(
    const arma::sp_mat& Y,
    const arma::mat& Yhat,
    const arma::vec& ybar,
    int threads
) {

  const arma::sp_mat Yt = Y.t();

  const arma::uword n_obs = Yt.n_cols;

  NumericVector result(2);
  arma::vec SSE(n_obs, arma::fill::zeros);
  arma::vec SST(n_obs, arma::fill::zeros);

  const double sum_ybar_sq = arma::dot(ybar, ybar);

  RcppThread::parallelFor(
    0,
    n_obs,
    [&Yt, &Yhat, &ybar, sum_ybar_sq, &SSE, &SST] (unsigned int d) {
      RcppThread::checkUserInterrupt();

      const arma::rowvec yhat_d = Yhat.row(d);

      double sse = arma::dot(yhat_d, yhat_d);
      double sst = sum_ybar_sq;

      for (arma::sp_mat::const_col_iterator it = Yt.begin_col(d);
           it != Yt.end_col(d); ++it) {

        const arma::uword v = it.row();
        const double y = (*it);
        const double yhat = yhat_d[v];

        sse += -(yhat * yhat) + (y - yhat) * (y - yhat);
        sst += -(ybar[v] * ybar[v]) + (y - ybar[v]) * (y - ybar[v]);
      }

      SSE(d) = sse;
      SST(d) = sst;
    },
    threads);

  result[ 0 ] = sum(SSE);
  result[ 1 ] = sum(SST);

  return result;

}
