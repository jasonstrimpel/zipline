//
//  optlib.c
//
//  Created by Jason Strimpel on 8/2/15.
//  Copyright (c) 2015 Jason Strimpel. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include <vector>
#include "stats.h"

extern "C" {

    double SIGN(double a, double b) {
        if (b>=0.0) {
            return fabsf(a);
        } else {
            return -fabsf(a);
        }
    }

    //
    // Black Scholes option pricing library
    //

    //
    // call options
    //

    // call option value
    double black_scholes_call_value(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt)+0.5*vol*time_sqrt;
        double d2 = d1-(vol*time_sqrt);
        return s*N(d1) - k*exp(-r*t)*N(d2);
    }

    // call option delta
    double black_scholes_call_delta(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        return N(d1);
    }

    // call option gamma
    double black_scholes_call_gamma(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        return n(d1)/(s*vol*time_sqrt);
    }

    // call option theta
    double black_scholes_call_theta(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        double d2 = d1-(vol*time_sqrt);
        return (-(s*vol*n(d1))/(2*time_sqrt)-r*k*exp(-r*t)*N(d2))/365.0;
    }

    // call option vega
    double black_scholes_call_vega(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        return (s*time_sqrt*n(d1))/100.0;
    }

    // call option rho
    double black_scholes_call_rho(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        double d2 = d1-(vol*time_sqrt);
        return (k*t*exp(-r*t)*N(d2))/100.0;
    }

    // objective function for implied volatility solver
    double black_scholes_call_iv_obj_function(double s, double k, double r, double t, double vol, double call_option_price) {
        return call_option_price-black_scholes_call_value(s,k,r,t,vol);
    }

    // brent solver for implied volatility
    double black_scholes_call_implied_volatility_brent(double s, double k, double r, double t, double call_option_price, double x1, double x2, double tol) {
        
        int ITMAX=100; // Maximum allowed number of iterations.
        double EPS=3.0e-8; // Machine floating-point precision.
        
        int iter;
        double a=x1,b=x2,c=x2,d=0.0,e=0.0,min1,min2;
        double fa=black_scholes_call_iv_obj_function(s,k,r,t,a,call_option_price);
        double fb=black_scholes_call_iv_obj_function(s,k,r,t,b,call_option_price);
        
        double fc,p,q,r_,s_,tol1,xm;
        if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
            //nrerror("Root must be bracketed in zbrent");
            return -400.0;
        fc=fb;
        for (iter=1;iter<=ITMAX;iter++) {
            if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
                c=a; // Rename a, b, c and adjust bounding interval d
                fc=fa;
                e=d=b-a;
            }
            if (fabsf(fc) < fabsf(fb)) {
                a=b;
                b=c;
                c=a;
                fa=fb;
                fb=fc;
                fc=fa;
            }
            tol1=2.0*EPS*fabsf(b)+0.5*tol; // Convergence check.
            xm=0.5*(c-b);
            if (fabsf(xm) <= tol1 || fb == 0.0) return b;
            if (fabsf(e) >= tol1 && fabsf(fa) > fabsf(fb)) {
                s_=fb/fa; // Attempt inverse quadratic interpolation.
                if (a == c) {
                    p=2.0*xm*s_;
                    q=1.0-s_;
                } else {
                    q=fa/fc;
                    r_=fb/fc;
                    p=s_*(2.0*xm*q*(q-r_)-(b-a)*(r_-1.0));
                    q=(q-1.0)*(r_-1.0)*(s_-1.0);
                }
                if (p > 0.0) q = -q; // Check whether in bounds.
                p=fabsf(p);
                min1=3.0*xm*q-fabsf(tol1*q);
                min2=fabsf(e*q);
                if (2.0*p < (min1 < min2 ? min1 : min2)) {
                    e=d; // Accept interpolation.
                    d=p/q;
                } else {
                    d=xm; // Interpolation failed, use bisection.
                    e=d;
                }
            } else { // Bounds decreasing too slowly, use bisection.
                d=xm;
                e=d;
            }
            a=b; // Move last best guess to a.
            fa=fb;
            if (fabsf(d) > tol1) {// Evaluate new trial root.
                b += d;
            } else {
                b += SIGN(tol1,xm);
            }
            fb=black_scholes_call_iv_obj_function(s,k,r,t,b,call_option_price);
        }
        //nrerror("Maximum number of iterations exceeded in zbrent");
        return -200.0; // Never get here.
    }




    //
    // put options
    //

    // put option value
    double black_scholes_put_value(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        double d2 = d1-(vol*time_sqrt);
        return k*exp(-r*t)*N(-d2) - s*N(-d1);
    }

    // put option delta
    double black_scholes_put_delta(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        return -N(-d1);
    }

    // put option gamma
    double black_scholes_put_gamma(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        return n(d1)/(s*vol*time_sqrt);
    }

    // put option theta
    double black_scholes_put_theta(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        double d2 = d1-(vol*time_sqrt);
        return (-(s*vol*n(d1)) / (2*time_sqrt)+ r*k * exp(-r*t) * N(-d2))/365.0;
    }

    // put option vega
    double black_scholes_put_vega(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        return (s * time_sqrt * n(d1))/100.0;
    }

    // put option rho
    double black_scholes_put_rho(double s, double k, double r, double t, double vol) {
        double time_sqrt = sqrt(t);
        double d1 = (log(s/k)+r*t)/(vol*time_sqrt) + 0.5*vol*time_sqrt;
        double d2 = d1-(vol*time_sqrt);
        return (-k*t*exp(-r*t) * N(-d2))/100.0;
    }

    // objective function for implied volatility solver
    double black_scholes_put_iv_obj_function(double s, double k, double r, double t, double vol, double call_option_price) {
        return call_option_price-black_scholes_put_value(s,k,r,t,vol);
    }

    // brent solver for implied volatility
    double black_scholes_put_implied_volatility_brent(double s, double k, double r, double t, double call_option_price, double x1, double x2, double tol) {
        
        int ITMAX=100; // Maximum allowed number of iterations.
        double EPS=3.0e-8; // Machine floating-point precision.
        
        int iter;
        double a=x1,b=x2,c=x2,d=0.0,e=0.0,min1,min2;
        double fa=black_scholes_put_iv_obj_function(s,k,r,t,a,call_option_price);
        double fb=black_scholes_put_iv_obj_function(s,k,r,t,b,call_option_price);
        
        double fc,p,q,r_,s_,tol1,xm;
        if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
            //nrerror("Root must be bracketed in zbrent");
            return -400.0;
        fc=fb;
        for (iter=1;iter<=ITMAX;iter++) {
            if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
                c=a; // Rename a, b, c and adjust bounding interval d
                fc=fa;
                e=d=b-a;
            }
            if (fabsf(fc) < fabsf(fb)) {
                a=b;
                b=c;
                c=a;
                fa=fb;
                fb=fc;
                fc=fa;
            }
            tol1=2.0*EPS*fabsf(b)+0.5*tol; // Convergence check.
            xm=0.5*(c-b);
            if (fabsf(xm) <= tol1 || fb == 0.0) return b;
            if (fabsf(e) >= tol1 && fabsf(fa) > fabsf(fb)) {
                s_=fb/fa; // Attempt inverse quadratic interpolation.
                if (a == c) {
                    p=2.0*xm*s_;
                    q=1.0-s_;
                } else {
                    q=fa/fc;
                    r_=fb/fc;
                    p=s_*(2.0*xm*q*(q-r_)-(b-a)*(r_-1.0));
                    q=(q-1.0)*(r_-1.0)*(s_-1.0);
                }
                if (p > 0.0) q = -q; // Check whether in bounds.
                p=fabsf(p);
                min1=3.0*xm*q-fabsf(tol1*q);
                min2=fabsf(e*q);
                if (2.0*p < (min1 < min2 ? min1 : min2)) {
                    e=d; // Accept interpolation.
                    d=p/q;
                } else {
                    d=xm; // Interpolation failed, use bisection.
                    e=d;
                }
            } else { // Bounds decreasing too slowly, use bisection.
                d=xm;
                e=d;
            }
            a=b; // Move last best guess to a.
            fa=fb;
            if (fabsf(d) > tol1) {// Evaluate new trial root.
                b += d;
            } else {
                b += SIGN(tol1,xm);
            }
            fb=black_scholes_put_iv_obj_function(s,k,r,t,b,call_option_price);
        }
        //nrerror("Maximum number of iterations exceeded in zbrent");
        return -200.0; // Never get here.
    }


    
    
    
    // call option
    double binomial_american_call_value(double s, double k, double r, double t, double vol, int steps) {
        double R = exp(r*(t/steps));
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(t/steps));
        double d = 1.0/u;
        double p_up = (R-d)/(u-d);
        double p_down = 1.0-p_up;
        
        std::vector<double> prices(steps+1);       // price of underlying
        prices[0] = s*pow(d, steps);  // fill in the endnodes.
        double uu = u*u;
        for (int i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        
        std::vector<double> call_values(steps+1);       // value of corresponding call
        for (int i=0; i<=steps; ++i) call_values[i] = fmax(0.0, (prices[i]-k)); // call payoffs at maturity
        
        for (int step=steps-1; step>=0; --step) {
            for (int i=0; i<=step; ++i) {
                call_values[i] = (p_up*call_values[i+1]+p_down*call_values[i])*Rinv;
                prices[i] = d*prices[i+1];
                call_values[i] = fmax(call_values[i],prices[i]-k);       // check for exercise
            };
        };
        return call_values[0];
    };
    
    
    
    
    double binomial_american_call_delta(double s, double k, double r, double t, double vol, int steps) {
        
        double R = exp(r*(t/steps));
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(t/steps));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        
        std::vector<double> prices (steps+1);
        prices[0] = s*pow(d, steps);
        for (int i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        
        std::vector<double> call_values (steps+1);
        for (int i=0; i<=steps; ++i) call_values[i] = fmax(0.0, (prices[i]-k));
        
        for (int CurrStep=steps-1 ; CurrStep>=1; --CurrStep) {
            for (int i=0; i<=CurrStep; ++i)   {
                prices[i] = d*prices[i+1];
                call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
                call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
            };
        };
        return (call_values[1]-call_values[0])/(s*u-s*d);
    };
    
    
    double binomial_american_call_gamma(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices(steps+1);
        std::vector<double> call_values(steps+1);
        double delta_t =(t/steps);
        double R = exp(r*delta_t);
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(delta_t));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        for (int i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (int i=0; i<=steps; ++i) call_values[i] = fmax(0.0, (prices[i]-k));
        for (int CurrStep=steps-1; CurrStep>=2; --CurrStep) {
            for (int i=0; i<=CurrStep; ++i)   {
                prices[i] = d*prices[i+1];
                call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
                call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
            };
        };
        double f22 = call_values[2];
        double f21 = call_values[1];
        double f20 = call_values[0];
        for (int i=0;i<=1;i++) {
            prices[i] = d*prices[i+1];
            call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
            call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
        };
        prices[0] = d*prices[1];
        call_values[0] = (pDown*call_values[0]+pUp*call_values[1])*Rinv;
        call_values[0] = fmax(call_values[0], s-k);        // check for exercise on first date
        double h = 0.5 * s * ( uu - d*d);
        return ( (f22-f21)/(s*(uu-1)) - (f21-f20)/(s*(1-d*d)) ) / h;
    };
    
    double binomial_american_call_theta(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices(steps+1);
        std::vector<double> call_values(steps+1);
        double delta_t =(t/steps);
        double R = exp(r*delta_t);
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(delta_t));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        for (int i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (int i=0; i<=steps; ++i) call_values[i] = fmax(0.0, (prices[i]-k));
        for (int CurrStep=steps-1; CurrStep>=2; --CurrStep) {
            for (int i=0; i<=CurrStep; ++i)   {
                prices[i] = d*prices[i+1];
                call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
                call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
            };
        };
        
        double f21 = call_values[1];
        
        for (int i=0;i<=1;i++) {
            prices[i] = d*prices[i+1];
            call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
            call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
        };
        
        prices[0] = d*prices[1];
        call_values[0] = (pDown*call_values[0]+pUp*call_values[1])*Rinv;
        call_values[0] = fmax(call_values[0], s-k);        // check for exercise on first date
        double f00 = call_values[0];
        
        return ((f21-f00) / (2*delta_t))/365.0;
        
    };
    
    double binomial_american_call_vega(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices(steps+1);
        std::vector<double> call_values(steps+1);
        double delta_t =(t/steps);
        double R = exp(r*delta_t);
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(delta_t));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        for (int i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (int i=0; i<=steps; ++i) call_values[i] = fmax(0.0, (prices[i]-k));
        for (int CurrStep=steps-1; CurrStep>=2; --CurrStep) {
            for (int i=0; i<=CurrStep; ++i)   {
                prices[i] = d*prices[i+1];
                call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
                call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
            };
        };
        
        for (int i=0;i<=1;i++) {
            prices[i] = d*prices[i+1];
            call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
            call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
        };
        
        prices[0] = d*prices[1];
        call_values[0] = (pDown*call_values[0]+pUp*call_values[1])*Rinv;
        call_values[0] = fmax(call_values[0], s-k);        // check for exercise on first date
        double f00 = call_values[0];
        
        double diff = 0.02;
        double tmp_sigma = vol+diff;
        double tmp_prices = binomial_american_call_value(s,k,r,t,tmp_sigma,steps);
        return ((tmp_prices-f00)/diff)/100.0;
    };
    
    double binomial_american_call_rho(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices(steps+1);
        std::vector<double> call_values(steps+1);
        double delta_t =(t/steps);
        double R = exp(r*delta_t);
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(delta_t));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        for (int i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (int i=0; i<=steps; ++i) call_values[i] = fmax(0.0, (prices[i]-k));
        for (int CurrStep=steps-1; CurrStep>=2; --CurrStep) {
            for (int i=0; i<=CurrStep; ++i)   {
                prices[i] = d*prices[i+1];
                call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
                call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
            };
        };
        
        for (int i=0;i<=1;i++) {
            prices[i] = d*prices[i+1];
            call_values[i] = (pDown*call_values[i]+pUp*call_values[i+1])*Rinv;
            call_values[i] = fmax(call_values[i], prices[i]-k);        // check for exercise
        };
        
        prices[0] = d*prices[1];
        call_values[0] = (pDown*call_values[0]+pUp*call_values[1])*Rinv;
        call_values[0] = fmax(call_values[0], s-k);        // check for exercise on first date
        double f00 = call_values[0];
        
        double diff = 0.02;
        double tmp_sigma = vol+diff;
        double tmp_prices = binomial_american_call_value(s,k,r,t,tmp_sigma,steps);
        
        diff = 0.05;
        double tmp_r = r+diff;
        tmp_prices = binomial_american_call_value(s,k,tmp_r,t,vol,steps);
        return ((tmp_prices-f00)/diff)/100.0;
    };
    
    
    
    // objective function for implied volatility solver
    double binomial_american_call_iv_obj_function(double s, double k, double r, double t, double vol, int steps, double call_option_price) {
        return call_option_price-binomial_american_call_value(s,k,r,t,vol,steps);
    }
    
    // brent solver for implied volatility
    // s,k,r,t,call_option_price,steps,x1,x2,tol
    double binomial_american_call_implied_volatility_brent(double s, double k, double r, double t, int steps, double call_option_price, double x1, double x2, double tol) {
        
        int ITMAX=100; // Maximum allowed number of iterations.
        double EPS=3.0e-8; // Machine floating-point precision.
        
        int iter;
        double a=x1,b=x2,c=x2,d=0.0,e=0.0,min1,min2;
        double fa=-binomial_american_call_iv_obj_function(s,k,r,t,a,steps,call_option_price);
        double fb=binomial_american_call_iv_obj_function(s,k,r,t,b,steps,call_option_price);
        
        double fc,p,q,r_,s_,tol1,xm;
        if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
            //nrerror("Root must be bracketed in zbrent");
            return -400.0;
        fc=fb;
        for (iter=1;iter<=ITMAX;iter++) {
            if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
                c=a; // Rename a, b, c and adjust bounding interval d
                fc=fa;
                e=d=b-a;
            }
            if (fabsf(fc) < fabsf(fb)) {
                a=b;
                b=c;
                c=a;
                fa=fb;
                fb=fc;
                fc=fa;
            }
            tol1=2.0*EPS*fabsf(b)+0.5*tol; // Convergence check.
            xm=0.5*(c-b);
            if (fabsf(xm) <= tol1 || fb == 0.0) return b;
            if (fabsf(e) >= tol1 && fabsf(fa) > fabsf(fb)) {
                s_=fb/fa; // Attempt inverse quadratic interpolation.
                if (a == c) {
                    p=2.0*xm*s_;
                    q=1.0-s_;
                } else {
                    q=fa/fc;
                    r_=fb/fc;
                    p=s_*(2.0*xm*q*(q-r_)-(b-a)*(r_-1.0));
                    q=(q-1.0)*(r_-1.0)*(s_-1.0);
                }
                if (p > 0.0) q = -q; // Check whether in bounds.
                p=fabsf(p);
                min1=3.0*xm*q-fabsf(tol1*q);
                min2=fabsf(e*q);
                if (2.0*p < (min1 < min2 ? min1 : min2)) {
                    e=d; // Accept interpolation.
                    d=p/q;
                } else {
                    d=xm; // Interpolation failed, use bisection.
                    e=d;
                }
            } else { // Bounds decreasing too slowly, use bisection.
                d=xm;
                e=d;
            }
            a=b; // Move last best guess to a.
            fa=fb;
            if (fabsf(d) > tol1) {// Evaluate new trial root.
                b += d;
            } else {
                b += SIGN(tol1,xm);
            }
            fb=binomial_american_call_iv_obj_function(s,k,r,t,b,steps,call_option_price);
        }
        //nrerror("Maximum number of iterations exceeded in zbrent");
        return -200.0; // Never get here.
    }
    
    
    
    
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // binomial put options
    ///////////////////////////////////////////////////////////////////////////////////////
    
    double binomial_american_put_value(double s, double k, double r, double t, double vol, int steps) {  // no steps in binomial tree
        double R = exp(r*(t/steps));            // interest rate for each step
        double Rinv = 1.0/R;                    // inverse of interest rate
        double u = exp(vol*sqrt(t/steps));    // up movement
        double uu = u*u;
        double d = 1.0/u;
        double p_up = (R-d)/(u-d);
        double p_down = 1.0-p_up;
        std::vector<double> prices(steps+1);       // price of underlying
        prices[0] = s*pow(d, steps);
        for (int i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        
        std::vector<double> put_values(steps+1);       // value of corresponding put
        for (int i=0; i<=steps; ++i) put_values[i] = fmax(0.0, (k-prices[i])); // put payoffs at maturity
        
        for (int step=steps-1; step>=0; --step) {
            for (int i=0; i<=step; ++i) {
                put_values[i] = (p_up*put_values[i+1]+p_down*put_values[i])*Rinv;
                prices[i] = d*prices[i+1];
                put_values[i] = fmax(put_values[i],(k-prices[i]));    // check for exercise
            };
        };
        return put_values[0];
    };
    
    double binomial_american_put_delta(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices (steps+1);
        std::vector<double> put_values  (steps+1);
        double R = exp(r*(t/steps));
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(t/steps));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        int i;
        for (i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (i=0; i<=steps; ++i) put_values[i] = fmax(0.0, (k - prices[i]));
        for (int CurrStep=steps-1 ; CurrStep>=1; --CurrStep) {
            for (i=0; i<=CurrStep; ++i)   {
                prices[i] = d*prices[i+1];
                put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
                put_values[i] = fmax(put_values[i], k-prices[i]);        // check for exercise
            };
        };
        return (put_values[1]-put_values[0])/(s*u-s*d);
    };
    
    double binomial_american_put_gamma(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices(steps+1);
        std::vector<double> put_values(steps+1);
        double delta_t =(t/steps);
        double R = exp(r*delta_t);
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(delta_t));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        int i;
        for (i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (i=0; i<=steps; ++i) put_values[i] = fmax(0.0, (k-prices[i]));
        for (int CurrStep=steps-1 ; CurrStep>=2; --CurrStep){
            for (i=0; i<=CurrStep; ++i){
                prices[i] = d*prices[i+1];
                put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
                put_values[i] = fmax(put_values[i], k-prices[i]); // check for exercise
            };
        };
        double f22 = put_values[2];
        double f21 = put_values[1];
        double f20 = put_values[0];
        for (i=0;i<=1;i++) {
            prices[i] = d*prices[i+1];
            put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
            put_values[i] = fmax(put_values[i], k-prices[i]); // check for exercise
        };
        
        prices[0] = d*prices[1];
        put_values[0] = (pDown*put_values[0]+pUp*put_values[1])*Rinv;
        put_values[0] = fmax(put_values[0], k-prices[i]); // check for exercise
        
        double h = 0.5 * s *( uu - d*d);
        return ( (f22-f21)/(s*(uu-1.0)) - (f21-f20)/(s*(1.0-d*d)) ) / h;
        
    };
    
    double binomial_american_put_theta(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices(steps+1);
        std::vector<double> put_values(steps+1);
        double delta_t =(t/steps);
        double R = exp(r*delta_t);
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(delta_t));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        int i;
        for (i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (i=0; i<=steps; ++i) put_values[i] = fmax(0.0, (k-prices[i]));
        for (int CurrStep=steps-1 ; CurrStep>=2; --CurrStep){
            for (i=0; i<=CurrStep; ++i){
                prices[i] = d*prices[i+1];
                put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
                put_values[i] = fmax(put_values[i], k-prices[i]); // check for exercise
            };
        };
        double f21 = put_values[1];
        for (i=0;i<=1;i++) {
            prices[i] = d*prices[i+1];
            put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
            put_values[i] = fmax(put_values[i], k-prices[i]); // check for exercise
        };
        
        prices[0] = d*prices[1];
        put_values[0] = (pDown*put_values[0]+pUp*put_values[1])*Rinv;
        put_values[0] = fmax(put_values[0], k-prices[i]); // check for exercise
        double f00 = put_values[0];
        
        return ((f21-f00) / (2*delta_t))/365.0;
        
    };
    
    double binomial_american_put_vega(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices(steps+1);
        std::vector<double> put_values(steps+1);
        double delta_t =(t/steps);
        double R = exp(r*delta_t);
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(delta_t));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        int i;
        for (i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (i=0; i<=steps; ++i) put_values[i] = fmax(0.0, (k-prices[i]));
        for (int CurrStep=steps-1 ; CurrStep>=2; --CurrStep){
            for (i=0; i<=CurrStep; ++i){
                prices[i] = d*prices[i+1];
                put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
                put_values[i] = fmax(put_values[i], k-prices[i]); // check for exercise
            };
        };
        
        for (i=0;i<=1;i++) {
            prices[i] = d*prices[i+1];
            put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
            put_values[i] = fmax(put_values[i], k-prices[i]); // check for exercise
        };
        
        prices[0] = d*prices[1];
        put_values[0] = (pDown*put_values[0]+pUp*put_values[1])*Rinv;
        put_values[0] = fmax(put_values[0], k-prices[i]); // check for exercise
        double f00 = put_values[0];
        
        double diff = 0.02;
        double tmp_sigma = vol+diff;
        double tmp_prices = binomial_american_put_value(s,k,r,t,tmp_sigma,steps);
        return ((tmp_prices-f00)/diff)/100.0;
        
    };
    
    double binomial_american_put_rho(double s, double k, double r, double t, double vol, int steps) {
        std::vector<double> prices(steps+1);
        std::vector<double> put_values(steps+1);
        double delta_t =(t/steps);
        double R = exp(r*delta_t);
        double Rinv = 1.0/R;
        double u = exp(vol*sqrt(delta_t));
        double d = 1.0/u;
        double uu= u*u;
        double pUp   = (R-d)/(u-d);
        double pDown = 1.0 - pUp;
        prices[0] = s*pow(d, steps);
        int i;
        for (i=1; i<=steps; ++i) prices[i] = uu*prices[i-1];
        for (i=0; i<=steps; ++i) put_values[i] = fmax(0.0, (k-prices[i]));
        for (int CurrStep=steps-1 ; CurrStep>=2; --CurrStep){
            for (i=0; i<=CurrStep; ++i){
                prices[i] = d*prices[i+1];
                put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
                put_values[i] = fmax(put_values[i], k-prices[i]); // check for exercise
            };
        };
        
        for (i=0;i<=1;i++) {
            prices[i] = d*prices[i+1];
            put_values[i] = (pDown*put_values[i]+pUp*put_values[i+1])*Rinv;
            put_values[i] = fmax(put_values[i], k-prices[i]); // check for exercise
        };
        
        prices[0] = d*prices[1];
        put_values[0] = (pDown*put_values[0]+pUp*put_values[1])*Rinv;
        put_values[0] = fmax(put_values[0], k-prices[i]); // check for exercise
        double f00 = put_values[0];
        
        double diff = 0.02;
        double tmp_sigma = vol+diff;
        double tmp_prices = binomial_american_put_value(s,k,r,t,tmp_sigma,steps);
        
        diff = 0.05;
        double tmp_r = r+diff;
        tmp_prices = binomial_american_put_value(s,k,tmp_r,t,vol,steps);
        return ((tmp_prices-f00)/diff)/100.0;
    };
    
    
    // objective function for implied volatility solver
    double binomial_american_put_iv_obj_function(double s, double k, double r, double t, double vol, int steps, double put_option_price) {
        return put_option_price-binomial_american_put_value(s,k,r,t,vol,steps);
    }
    
    // brent solver for implied volatility
    // s,k,r,t,call_option_price,steps,x1,x2,tol
    double binomial_american_put_implied_volatility_brent(double s, double k, double r, double t, int steps, double put_option_price, double x1, double x2, double tol) {
        
        int ITMAX=100; // Maximum allowed number of iterations.
        double EPS=3.0e-8; // Machine floating-point precision.
        
        int iter;
        double a=x1,b=x2,c=x2,d=0.0,e=0.0,min1,min2;
        double fa=-binomial_american_call_iv_obj_function(s,k,r,t,a,steps,put_option_price);
        double fb=binomial_american_call_iv_obj_function(s,k,r,t,b,steps,put_option_price);
        
        double fc,p,q,r_,s_,tol1,xm;
        if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
            //nrerror("Root must be bracketed in zbrent");
            return -400.0;
        fc=fb;
        for (iter=1;iter<=ITMAX;iter++) {
            if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
                c=a; // Rename a, b, c and adjust bounding interval d
                fc=fa;
                e=d=b-a;
            }
            if (fabsf(fc) < fabsf(fb)) {
                a=b;
                b=c;
                c=a;
                fa=fb;
                fb=fc;
                fc=fa;
            }
            tol1=2.0*EPS*fabsf(b)+0.5*tol; // Convergence check.
            xm=0.5*(c-b);
            if (fabsf(xm) <= tol1 || fb == 0.0) return b;
            if (fabsf(e) >= tol1 && fabsf(fa) > fabsf(fb)) {
                s_=fb/fa; // Attempt inverse quadratic interpolation.
                if (a == c) {
                    p=2.0*xm*s_;
                    q=1.0-s_;
                } else {
                    q=fa/fc;
                    r_=fb/fc;
                    p=s_*(2.0*xm*q*(q-r_)-(b-a)*(r_-1.0));
                    q=(q-1.0)*(r_-1.0)*(s_-1.0);
                }
                if (p > 0.0) q = -q; // Check whether in bounds.
                p=fabsf(p);
                min1=3.0*xm*q-fabsf(tol1*q);
                min2=fabsf(e*q);
                if (2.0*p < (min1 < min2 ? min1 : min2)) {
                    e=d; // Accept interpolation.
                    d=p/q;
                } else {
                    d=xm; // Interpolation failed, use bisection.
                    e=d;
                }
            } else { // Bounds decreasing too slowly, use bisection.
                d=xm;
                e=d;
            }
            a=b; // Move last best guess to a.
            fa=fb;
            if (fabsf(d) > tol1) {// Evaluate new trial root.
                b += d;
            } else {
                b += SIGN(tol1,xm);
            }
            fb=binomial_american_put_iv_obj_function(s,k,r,t,b,steps,put_option_price);
        }
        //nrerror("Maximum number of iterations exceeded in zbrent");
        return -200.0; // Never get here.
    }

    
    
    
    
    
    
    
    
    
    
    
};
