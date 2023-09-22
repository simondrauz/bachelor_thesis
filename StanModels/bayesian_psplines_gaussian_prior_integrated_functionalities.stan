functions {
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order);
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order) {
    //
    // DOCSTRING:
    //    The implementation of spline matrices in Stan is mainly based on the implementation of
    //    Splines in Stan by Milad Kharratzadeh (https://mc-stan.org/users/documentation/case-studies/splines_in_stan.html)
    //    The function has been verified with the bs() function from the spline function such that
    //    our goal is to mimic a call of bs(knots=interior_knots, degree=spline_degree, intercept=FALSE)
    // INPUTS:
    //    t:          the points at which the b_spline is calculated
    //    ext_knots:  the set of extended knots
    //    ind:        the index of the b_spline
    //    order:      the order of the b-spline
    //
    vector[size(t)] b_spline;
    vector[size(t)] w1 = rep_vector(0, size(t));
    vector[size(t)] w2 = rep_vector(0, size(t));
    if (order==1)
      for (i in 1:size(t)) // B-splines of order 1 are piece-wise constant
    b_spline[i] = (ext_knots[ind] <= t[i]) && (t[i] < ext_knots[ind+1]);
    else {
      if (ext_knots[ind] != ext_knots[ind+order-1])
        w1 = (to_vector(t) - rep_vector(ext_knots[ind], size(t))) /
          (ext_knots[ind+order-1] - ext_knots[ind]);
      if (ext_knots[ind+1] != ext_knots[ind+order])
        w2 = 1 - (to_vector(t) - rep_vector(ext_knots[ind+1], size(t))) /
          (ext_knots[ind+order] - ext_knots[ind+1]);
      // Calculating the B-spline recursively as linear interpolation of two lower-order splines
      b_spline = w1 .* build_b_spline(t, ext_knots, ind, order-1) +
        w2 .* build_b_spline(t, ext_knots, ind+1, order-1);
    }
    return b_spline;
  }

  vector generate_knots(real[] covariate_data, int no_knots_extended);
  vector generate_knots(real[] covariate_data, int no_knots_extended) {
    //
    // DOCSTRING:
    //    Define function which generates a vector with interior and exterior knots, based on the
    //    desired number of (exterior+interior knots) and equal-spacing approach
    //
    // INPUTS:
    //    covariate_data:     data corresponding to covariate
    //    no_knots_extended:  number of knots including exterior knots
    //
    vector[no_knots_extended] knots;
    real min_covariate = min(covariate_data);
    real max_covariate = max(covariate_data);
    real spacing = (max_covariate - min_covariate) / (no_knots_extended -1);

    print("Generate knot list for a covariate...");

    for (i in 1:(no_knots_extended)) {
      knots[i] = min_covariate + (i - 1) * spacing;
    }
    return knots;
  }

  matrix generate_spline_basis_matrix(real[] covariate_data, int spline_degree, int no_interior_knots,  int no_data, int no_basis);
  matrix generate_spline_basis_matrix(real[] covariate_data, int spline_degree, int no_interior_knots,  int no_data, int no_basis) {
    //
    // DOCSTRING:
    //    Define function to generate spline design matrix: The knots passed to the function comprise
    //    interior as well as exterior knots.
    //    The result of the function resembles a call of the bs() function with:
    //       knots = knots[-c(-1, length(knots))] = interior_knots
    //       degree = spline_degree
    //       intercept = FALSE
    //    The approach is to generate a design matrix including an intercept and dropping it
    //    afterwards to get the design matrix without intercept
    //
    // INPUTS:
    //    covariate_data:       data corresponding to covariate
    //    spline_degree:        the degree of spline (is equal to order - 1)
    //    no_interior_knots:    no of interior knots for covariate
    //    no_data:              number of data points in training data
    //    no_basis:             dimension of our basis matrix for covariate modelled without intercept
    //                          (equals no_interior_knots + spline_degree)
    //
    matrix[no_basis, no_data] B_without_icpt;
    matrix[no_data, no_basis] B_without_icpt_transposed; // to equal return dimensions of bs()
    matrix[(no_basis+1), no_data] B;
    int no_knots_extended = no_interior_knots + 2;
    vector[no_knots_extended] knots;
    vector[2 * spline_degree + no_knots_extended] ext_knots;
    vector[spline_degree + no_knots_extended] ext_knots_temp;

    print("Generate spline design matrix for a covariate...")
    knots = generate_knots(covariate_data, no_knots_extended);
    ext_knots_temp = append_row(rep_vector(knots[1], spline_degree), to_vector(knots));
    ext_knots = append_row(ext_knots_temp, rep_vector(knots[no_knots_extended], spline_degree));

    for (ind in 1:(no_basis+1)) {
      B[ind, :] = to_row_vector(build_b_spline(covariate_data, to_array_1d(ext_knots), ind, spline_degree + 1));
    }

    B[no_basis+1, no_data] = 1;
    B_without_icpt = B[2:(no_basis+1), :];
    B_without_icpt_transposed = B_without_icpt';

    return B_without_icpt_transposed;
    }

  matrix generate_penalty_matrix(int no_basis);
  matrix generate_penalty_matrix(int no_basis) {
    //
    // DOCSTRING:
    //    Define function which generates a first order random walk penalty matrix based on the basis of the spline
    //    design matrix
    //
    // INPUTS:
    //    no_basis:             dimension of our basis matrix for covariate modelled without intercept
    //                          (equals no_interior_knots + spline_degree)
    //
    matrix[no_basis, no_basis] penalty_matrix;
    print("Generate penalty matrix for a covariate...")
    // Initialize all entries to zero
    for (i in 1:no_basis) {
      for (j in 1:no_basis) {
        penalty_matrix[i, j] = 0;
      }
    }

    // Fill the main diagonal with 2s
    for (k in 1:no_basis) {
      penalty_matrix[k, k] = 2;
    }

    // Modify the [1,1] and [no_basis,no_basis] elements to be 1
    penalty_matrix[1, 1] = 1;
    penalty_matrix[no_basis, no_basis] = 1;

    // Add -1s to the sub-diagonals above and below the main diagonal
    for (l in 2:no_basis) {
      penalty_matrix[l, l - 1] = -1;
      penalty_matrix[l - 1, l] = -1;
    }

    return penalty_matrix;
  }
}

data {
    int<lower=1> no_data;                         // number of data points in training data
    int<lower=0> no_data_eval;                    // number of data points for evaluation
    int<lower=1> spline_degree;                   // the degree of spline (is equal to order - 1)

    real covariate_data_X1[no_data] ;             // data corresponding to covariate X1
    real covariate_data_X1_eval[no_data_eval];    // evaluation data corresponding to covariate X1
    int<lower=1> no_basis_X1;                     // dimension of our basis matrix for covariate X1 modelled without intercept
                                                  //(equals no_interior_knots + spline_degree)
    int<lower=1>no_interior_knots_X1;             // no of interior knots for covariate X1

    real covariate_data_X2[no_data];              // compare covariate_X1 for X2 respectivly
    real covariate_data_X2_eval[no_data_eval];    // compare covariate_X1 for X2 respectivly
    int<lower=1> no_basis_X2;                     // compare no_basis_X1 for X2 respectivly
    int<lower=1> no_interior_knots_X2;            // compare no_interior_knots_X1 for X2 respectivly

    int Y[no_data];                               // Observed Target variable
    int Y_eval[no_data_eval];                     // Observed Target variable for evaluation

    real a_tau_X1;                                // Shape parameter for InverseGamma of tau for X1
    real b_tau_X1;                                // Scale parameter for InverseGamma of tau for X1
    real a_tau_X2;                                // Shape parameter for InverseGamma of tau for X2
    real b_tau_X2;                                // Scale parameter for InverseGamma of tau for X2
    real a_alpha;                                 // Shape parameter for Gamma of alpha
    real b_alpha;                                 // Scale parameter for Gamma of alpha
    real intercept_mu;                            // Mean of intercept prior
    real<lower=0> intercept_sigma;                // Standard deviation of intercept prior

}

transformed data {
  // Intialize matrices to be generated
  matrix[no_data, no_basis_X1] basis_X1;
  matrix[no_data, no_basis_X2] basis_X2;
  matrix[no_data_eval, no_basis_X1] basis_X1_eval;
  matrix[no_data_eval, no_basis_X2] basis_X2_eval;
  matrix[no_basis_X1, no_basis_X1] penalty_matrix_X1;
  matrix[no_basis_X2, no_basis_X2] penalty_matrix_X2;

  // Generate design matrix of regression splines for covariate X1
  basis_X1 = generate_spline_basis_matrix(covariate_data_X1, spline_degree, no_interior_knots_X1, no_data, no_basis_X1);
  print("basis_X1 generated")
  // Generate design matrix of regression splines for covariate X2
  basis_X2 = generate_spline_basis_matrix(covariate_data_X2, spline_degree, no_interior_knots_X2, no_data, no_basis_X2);
  print("basis_X2 generated")

  // Generate design matrix of evaluation regression splines for covariate X1
  basis_X1_eval = generate_spline_basis_matrix(covariate_data_X1_eval, spline_degree, no_interior_knots_X1, no_data_eval, no_basis_X1);
  print("basis_X1_eval generated")

  // Generate design matrix of evaluation regression splines for covariate X2
  basis_X2_eval = generate_spline_basis_matrix(covariate_data_X2_eval, spline_degree, no_interior_knots_X2, no_data_eval, no_basis_X2);
  print("basis_X2_eval generated")

  // Generate penalty matrix for gaussian prior of covariate X1
  penalty_matrix_X1 = generate_penalty_matrix(no_basis_X1);
  print("penalty_matrix_X1 generated")
  // Generate penalty matrix for gaussian prior of covariate X2
  penalty_matrix_X2 = generate_penalty_matrix(no_basis_X2);
  print("penalty_matrix_X2 generated")
}

parameters {
    real intercept;                                                // regression intercept
    vector[no_basis_X1] spline_coefficients_X1;                    // Spline coefficients for X1
    vector[no_basis_X2] spline_coefficients_X2;                    // Spline coefficients for X2
    real<lower=0> tau_X1;                                          // Precision parameter for Gaussian random walk for X1
    real<lower=0> tau_X2;                                          // Precision parameter for Gaussian random walk for X2
    real<lower=0> alpha;                                           // Negative Binomial dispersion parameter
}

transformed parameters {
    vector[no_data] mu;                                                   // Mean of Negative Binomial distribution
    matrix[no_basis_X1, no_basis_X1] precision_matrix_X1;           // precision matrix for X1
    matrix[no_basis_X2, no_basis_X2] precision_matrix_X2;           // precision matrix for X2

    mu = exp(intercept + basis_X1 * spline_coefficients_X1 + basis_X2 * spline_coefficients_X2);
                                                                    //taking exponential as a link function of a GAM

    precision_matrix_X1 = penalty_matrix_X1 / pow(tau_X1, 2);       // Defining precision matrix for X1
    precision_matrix_X2 = penalty_matrix_X2 / pow(tau_X2, 2);       // Defining precision matrix for X2
}

model {
  // Priors
    intercept ~ normal(intercept_mu, intercept_sigma);               // Diffuse prior for intercept
    tau_X1 ~ inv_gamma(a_tau_X1, b_tau_X1);                           // Inverse Gamma prior for tau_X1
    tau_X2 ~ inv_gamma(a_tau_X2, b_tau_X2);                          // Inverse Gamma prior for tau_X2
    alpha ~ gamma(a_alpha, b_alpha);                                 // Gamma prior for alpha

    spline_coefficients_X1 ~ multi_normal_prec(rep_vector(0, no_basis_X1), precision_matrix_X1);
                                                                     // Prior for spline coefficients for X1
    spline_coefficients_X2 ~ multi_normal_prec(rep_vector(0, no_basis_X2), precision_matrix_X2);
                                                                     // Prior for spline coefficients for X2

    Y ~ neg_binomial_2(mu, alpha);                                   // Likelihood
}

generated quantities {
  int y_pred_train[no_data];
  int y_pred_eval[no_data_eval];
  vector[no_data_eval] mu_eval;

  mu_eval = exp(intercept + basis_X1_eval * spline_coefficients_X1 + basis_X2_eval * spline_coefficients_X2);
  for (n in 1:no_data) {
      y_pred_train[n] = neg_binomial_2_rng(mu[n], alpha);
  }
  for (n_eval in 1:no_data_eval) {
      y_pred_eval[n_eval] = neg_binomial_2_rng(mu_eval[n_eval], alpha);
  }
}
