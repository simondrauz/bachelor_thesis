data {
    int<lower=1> N;          // Number of data points
    int<lower=1> num_basis_X1;  // Number of coefficients for P-splines for X1
    int<lower=1> num_basis_X2;  // Number of coefficients for P-splines for X2

    matrix[N, num_basis_X1] basis_X1;  // Basis matrix for regressor X1
    matrix[N, num_basis_X2] basis_X2;  // Basis matrix for regressor X2
    int Y[N];             // Observed Target variable

    matrix[num_basis_X1, num_basis_X1] K_X1;  // Penalty matrix for X1
    matrix[num_basis_X2, num_basis_X2] K_X2;  // Penalty matrix for X2

    real a_tau_X1;              // Shape parameter for InverseGamma of tau for X1
    real b_tau_X1;              // Scale parameter for InverseGamma of tau for X1
    real a_tau_X2;            // Shape parameter for InverseGamma of tau for X2
    real b_tau_X2;            // Scale parameter for InverseGamma of tau for X2
    real a_alpha;             // Shape parameter for Gamma of alpha
    real b_alpha;             // Scale parameter for Gamma of alpha
    real intercept_mu;        // Mean of intercept prior
    real<lower=0> intercept_sigma;     // Standard deviation of intercept prior

    int<lower=0> N_eval;                 // Number of data points for evaluation
    matrix[N_eval, num_basis_X1] basis_X1_eval;    // Basis matrix for regressor X1_eval
    matrix[N_eval, num_basis_X2] basis_X2_eval;    // Basis matrix for regressor X2_eval
    int Y_eval[N_eval];                            // Observed Target variable for evaluation
}

parameters {
    real intercept;                           // regression intercept
    vector[num_basis_X1] spline_coefficients_X1; // Spline coefficients for X1
    vector[num_basis_X2] spline_coefficients_X2; // Spline coefficients for X2
    real<lower=0> tau_X1;                        // Precision parameter for Gaussian random walk for X1
    real<lower=0> tau_X2;                      // Precision parameter for Gaussian random walk for X2
    real<lower=0> alpha;                      // Negative Binomial dispersion parameter
}
transformed parameters {
    vector[N] mu; // Mean of Negative Binomial distribution
    matrix[num_basis_X1, num_basis_X1] precision_matrix_X1;  // precision matrix for X1
    matrix[num_basis_X2, num_basis_X2] precision_matrix_X2;  // precision matrix for X2

    mu = exp(intercept + basis_X1 * spline_coefficients_X1 + basis_X2 * spline_coefficients_X2); //taking exponential as a link function of a GAM

    precision_matrix_X1 = K_X1 / pow(tau_X1, 2);  // Defining precision matrix for X1
    precision_matrix_X2 = K_X2 / pow(tau_X2, 2);  // Defining precision matrix for X2
}

model {
    // Priors
    intercept ~ normal(intercept_mu, intercept_sigma);               // Diffuse prior for intercept
    tau_X1 ~ inv_gamma(a_tau_X1, b_tau_X1);  // Inverse Gamma prior for tau_X1
    tau_X2 ~ inv_gamma(a_tau_X2, b_tau_X2);  // Inverse Gamma prior for tau_X2
    alpha ~ gamma(a_alpha, b_alpha);              // Gamma prior for alpha

    spline_coefficients_X1 ~ multi_normal_prec(rep_vector(0, num_basis_X1), precision_matrix_X1);  // Prior for spline coefficients for X1
    spline_coefficients_X2 ~ multi_normal_prec(rep_vector(0, num_basis_X2), precision_matrix_X2);  // Prior for spline coefficients for X2

    // Likelihood
    Y ~ neg_binomial_2(mu, alpha);
}
generated quantities{
    int y_pred_train[N];
    int y_pred_eval[N_eval];
    vector[N_eval] mu_eval;

    mu_eval = exp(intercept + basis_X1_eval * spline_coefficients_X1 + basis_X2_eval * spline_coefficients_X2);
    for (n in 1:N) {
        y_pred_train[n] = neg_binomial_2_rng(mu[n], alpha);
    }
    for (n_eval in 1:N_eval) {
        y_pred_eval[n_eval] = neg_binomial_2_rng(mu_eval[n_eval], alpha);
    }
}
