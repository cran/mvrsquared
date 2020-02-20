
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#define ARMA_64BIT_WORD
using namespace Rcpp ;

// [[Rcpp::export]]
NumericVector calc_sum_squares_latent(arma::sp_mat Y, NumericMatrix X, NumericMatrix W, NumericVector ybar) {

  int n_obs = Y.n_rows; // number of observations
  int n_latent_vars = X.ncol(); // number of latent variables
  int n_explicit_vars = W.ncol(); // number of explicit variables
  NumericVector result(2); // final result
  double SSE = 0; // sum of squared errors across all documents
  double SST = 0; // total sum of squares across all documents


  // for each observations...
  for(int d = 0; d < n_obs; d++){

    R_CheckUserInterrupt();

    // Yhat = X %*% W. But doing it funny below to optimize calculation
    double sse = 0;
    double sst = 0;

    for(int v = 0; v < n_explicit_vars; v++ ){
      double Yhat = 0;

      for(int k = 0; k < n_latent_vars; k++ ){
        Yhat = Yhat + X(d , k ) * W(k , v );
      }

      sse = sse + ((Y(d , v) - Yhat) * (Y(d , v) - Yhat));

      sst = sst + ((Y(d , v) - ybar[ v ]) * (Y(d , v) - ybar[ v ]));

    }

    SSE = SSE + sse;

    SST = SST + sst;
  }

  result[ 0 ] = SSE;
  result[ 1 ] = SST;

  return result;


}
