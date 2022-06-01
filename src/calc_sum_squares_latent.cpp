
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::depends(RcppThread)]]
#include "RcppThread.h"
#include <cmath>

using namespace Rcpp ;

// [[Rcpp::export]]
NumericVector calc_sum_squares_latent(
    arma::sp_mat Y,
    arma::mat X,
    arma::mat W,
    arma::vec ybar,
    int threads
) {

  Y = Y.t(); // transpose Y to take advantage of column major for parallelism

  int n_obs = Y.n_cols; // number of observations
  NumericVector result(2); // final result
  arma::vec SSE(n_obs); // sum of squared errors across all documents
  arma::vec SST(n_obs); // total sum of squares across all documents

  // // for each observations...
  RcppThread::parallelFor(
    0,
    n_obs,
    [&Y,
     &X,
     &W,
     &ybar,
     &SSE,
     &SST
    ] (unsigned int d){
      RcppThread::checkUserInterrupt();

      // Yhat = X %*% W. But doing it funny below to optimize calculation
      double sse = 0;
      double sst = 0;

      for(int v = 0; v < W.n_cols; v++ ){
        double Yhat = 0;

        for(int k = 0; k < X.n_cols; k++ ){
          Yhat = Yhat + X(d , k ) * W(k , v );
        }

        sse = sse + ((Y(v, d) - Yhat) * (Y(v, d) - Yhat));

        sst = sst + ((Y(v, d) - ybar[ v ]) * (Y(v, d) - ybar[ v ]));

      }

      SSE(d) = sse;

      SST(d) = sst;
    },
    threads);

  result[ 0 ] = sum(SSE);
  result[ 1 ] = sum(SST);

  return result;


}
