//
//  optlib.h
//
//  Created by Jason Strimpel on 8/2/15.
//  Copyright (c) 2015 Jason Strimpel. All rights reserved.
//

#ifndef __optlib__
#define __optlib__

double black_scholes_call_value(double s, double k, double r, double t, double vol);
double black_scholes_call_delta(double s, double k, double r, double t, double vol);
double black_scholes_call_gamma(double s, double k, double r, double t, double vol);
double black_scholes_call_vega(double s, double k, double r, double t, double vol);
double black_scholes_call_theta(double s, double k, double r, double t, double vol);
double black_scholes_call_rho(double s, double k, double r, double t, double vol);
double black_scholes_call_implied_volatility_brent(double s, double k, double r, double t, double call_option_price, double x1, double x2, double tol);

double black_scholes_put_value(double s, double k, double r, double t, double vol);
double black_scholes_put_delta(double s, double k, double r, double t, double vol);
double black_scholes_put_gamma(double s, double k, double r, double t, double vol);
double black_scholes_put_vega(double s, double k, double r, double t, double vol);
double black_scholes_put_theta(double s, double k, double r, double t, double vol);
double black_scholes_put_rho(double s, double k, double r, double t, double vol);
double black_scholes_put_implied_volatility_brent(double s, double k, double r, double t, double call_option_price, double x1, double x2, double tol);

double binomial_american_call_value(double s, double k, double r, double t, double vol, int steps);
double binomial_american_call_delta(double s, double k, double r, double t, double vol, int steps);
double binomial_american_call_gamma(double s, double k, double r, double t, double vol, int steps);
double binomial_american_call_theta(double s, double k, double r, double t, double vol, int steps);
double binomial_american_call_vega(double s, double k, double r, double t, double vol, int steps);
double binomial_american_call_rho(double s, double k, double r, double t, double vol, int steps);
double binomial_american_call_implied_volatility_brent(double s, double k, double r, double t, double call_option_price, int steps, double x1, double x2, double tol);

double binomial_american_put_value(double s, double k, double r, double t, double vol, int steps);
double binomial_american_put_delta(double s, double k, double r, double t, double vol, int steps);
double binomial_american_put_gamma(double s, double k, double r, double t, double vol, int steps);
double binomial_american_put_theta(double s, double k, double r, double t, double vol, int steps);
double binomial_american_put_vega(double s, double k, double r, double t, double vol, int steps);
double binomial_american_put_rho(double s, double k, double r, double t, double vol, int steps);
double binomial_american_put_implied_volatility_brent(double s, double k, double r, double t, double call_option_price, int steps, double x1, double x2, double tol);

#endif /* defined(__optlib__) */
