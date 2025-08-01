import jax
from jax import random
import jax.numpy as np
from jax.scipy.linalg import cho_factor,cho_solve # necessary for Cholesky factorization
from jax import value_and_grad
# from scipy.optimize import curve_fit

jax.config.update("jax_enable_x64", True)

prng_key = random.key(0)

def init_prng(seed):
    global prng_key
    prng_key = random.PRNGKey(seed)
    return prng_key
#

def grab_prng():
    global prng_key
    _,prng_key = random.split(prng_key)
    return prng_key
#

# transform unconstrained hyperparameters to constrained (ensure strictly positive).
def param_transform(unconstrained_hyperparams):
    return np.exp(unconstrained_hyperparams)
#

# transform constrained hyperparameters to unconstrained
def inverse_param_transform(hyperparams):
    return np.log(hyperparams)
#

# evaluate the squared-exponential kernel between all pairs of points from X1 and X2, using kernel hyperparameters (hyperparams).
def sqexp_cov_function(X1, X2, hyperparams):
    noise_variance = hyperparams[0]
    signal_variance = hyperparams[1]
    l = hyperparams[2:]
    
    sqr_dis = np.sum( (np.expand_dims(X1, axis=1)-np.expand_dims(X2, axis=0)) ** 2, axis=2 ) # squared Euclidean distance matrix
    # return noise_variance * np.where(sqr_dis == 0, 1, 0) + signal_variance * np.exp(- sqr_dis / l)
    return signal_variance * np.exp(- sqr_dis / l)
#

# evaluate the Mahalanobis-based squared-exponential kernel between all pairs of points from X1 and X2, using kernel hyperparameters (hyperparams).
def sqexp_mahalanobis_cov_function(X1, X2, hyperparams):
    noise_variance = hyperparams[0]
    signal_variance = hyperparams[1]
    l = hyperparams[2:]
    
    L_sqrt_inv = np.diag(np.sqrt(1/l))
    X1_tilde = X1 @ L_sqrt_inv
    X2_tilde = X2 @ L_sqrt_inv
    sqr_dis = np.sum( (np.expand_dims(X1_tilde, 1)-np.expand_dims(X2_tilde, 0)) ** 2, axis=2 ) # squared Euclidean distance matrix
    return signal_variance * np.exp(- sqr_dis)
#

# compute the log marginal likelihood (LML)
def log_marginal_likelihood(cov_func, X_train, Y_train):
    N = X_train.shape[0]
    def lml_function(unconstrained_hyperparams):
        hyperparams = param_transform(unconstrained_hyperparams)
        noise_variance = hyperparams[0]
        signal_variance = hyperparams[1]
        l = hyperparams[2:]
        
        cov = noise_variance*np.eye(N) + cov_func(X_train, X_train, hyperparams)
        cfac = cho_factor(cov, lower=True)
        Lower = cfac[0]
        alpha = cho_solve(cfac, Y_train)
        # det = np.linalg.det(Lower)**2
        # return -0.5 * np.log(det) - 0.5 * np.transpose(Y_train) @ alpha
        
        # return -1 * np.log(np.linalg.det(Lower)) - 0.5 * np.transpose(Y_train) @ alpha
        
        return -1 * np.sum(np.log(np.diag(Lower))) - 0.5 * np.transpose(Y_train) @ alpha
    #
    return lml_function
#

# gaussian process
def gp_posterior(cov_func, X_train, Y_train, hyperparams):
    N = X_train.shape[0]
    noise_variance = hyperparams[0]
    signal_variance = hyperparams[1]
    l = hyperparams[2:]
    
    Kxx = noise_variance*np.eye(N) + cov_func(X_train, X_train, hyperparams)
    cfac = cho_factor(Kxx, lower=True)
    Lower = cfac[0]
    alpha = cho_solve(cfac, Y_train)
    def posterior_predictive(X_star):
        Kxstar = cov_func(X_train, X_star, hyperparams)
        Kstarstar = cov_func(X_star, X_star, hyperparams)
        posterior_mean = np.transpose(Kxstar) @ alpha
        posterior_var = Kstarstar - np.transpose(Kxstar) @ cho_solve(cfac, Kxstar)
        #
        diags = np.diag(posterior_var)
        posterior_var = np.diag(diags)
        #
        return (posterior_mean, posterior_var)
    #
    return posterior_predictive
#

# compute the negative log of the predictive density (NLPD)
def neg_log_predictive_density(Y_test, posterior_mean, posterior_var, noise_variance):
    D = Y_test.shape[0] # number of test samples
    miu_hat = posterior_mean
    SIGMA_hat = noise_variance * np.eye(D) + posterior_var
    dif = (Y_test-miu_hat)
    cfac = cho_factor(SIGMA_hat, lower=True)
    Lower = cfac[0]
    return 0.5 * (D*np.log(2*np.pi) + 2*np.sum(np.log(np.diag(Lower))) + np.transpose(dif)@cho_solve(cfac, dif))
#

# optimization loop based on LML
def empirical_bayes(cov_func, X_train, Y_train, unconstrained_hyperparams_init, step_size, T):
    lml_function = log_marginal_likelihood(cov_func, X_train, Y_train)
    val_grad_lml_function = value_and_grad(lml_function)
    unconstrained_hyperparams = unconstrained_hyperparams_init
    for i in range(T):
        val, grad = val_grad_lml_function(unconstrained_hyperparams)
        print(i, val)
        unconstrained_hyperparams += grad * step_size
    #
    # print(val)
    return (unconstrained_hyperparams, val)
#

# calculate the conditional log marginal likelihood (CLML)
def conditional_log_marginal_likelihood(cov_func, X_con, Y_con):
    X_star = X_con[1]
    Y_test = Y_con[1]

    def clml_function(unconstrained_hyperparams):
        hyperparams = param_transform(unconstrained_hyperparams)
        posterior_predictive = gp_posterior(cov_func, X_con[0], Y_con[0], hyperparams)
        (posterior_mean, posterior_var) = posterior_predictive(X_star)
        return -neg_log_predictive_density(Y_test, posterior_mean, posterior_var, hyperparams[0])
    
    return clml_function
#

# optimization loop based on CLML
def clml_opt(cov_func, X_con, Y_con, unconstrained_hyperparams_init, step_size, T):
    clml_function = conditional_log_marginal_likelihood(cov_func, X_con, Y_con)
    val_grad_function = value_and_grad(clml_function)
    unconstrained_hyperparams = unconstrained_hyperparams_init
    for i in range(T):
        val, grad = val_grad_function(unconstrained_hyperparams)
        print(i, val)
        unconstrained_hyperparams += grad * step_size
    #
    # print(val)
    return (unconstrained_hyperparams, val)
#

# def clmls_model(x, paras):
#     a,b,c,d = paras
#     return a - b * np.exp(-c * x + d)

# def mse(x_data, y_data, seq_len=8):

#     def mse_function(paras):
#         a,b,c,d = paras
#         y_pred = clmls_model(x_data, paras)
#         return np.sum((y_data - y_pred)**2)
#     #
#     return mse_function

# def opt_clmls(clml_vals, T=200, step_size=1e-4):
#     # Data setup
#     x_data = np.arange(len(clml_vals))  # x values (same as indices)
#     y_data = np.array(clml_vals)  # y values
#     para_init = np.array([clml_vals[-1], 1.0, 1.0, -1.0])

#     mse_function = mse(x_data, y_data)
#     val_grad_function = value_and_grad(mse_function)
#     para = para_init
#     for i in range(T):
#         val, grad = val_grad_function(para)
#         para -= grad * step_size
#     #
#     return para

# linear regression model to fit CLMLS
def opt_clmls(clml_vals):
    # Data setup
    x_data = np.arange(len(clml_vals))  # x values (same as indices)
    y_data = np.array(clml_vals)  # y values

    weights = np.arange(1, len(clml_vals) + 1)
    w_sum = np.sum(weights)
    x_mean = np.sum(weights * x_data) / w_sum
    y_mean = np.sum(weights * y_data) / w_sum

    # Weighted covariance and variance
    covariance = np.sum(weights * (x_data - x_mean) * (y_data - y_mean))
    variance = np.sum(weights * (x_data - x_mean) ** 2)

    # Compute slope (b) and intercept (a)
    b = covariance / variance
    a = y_mean - b * x_mean
    # print(a, b)
    return a, b

# calculate the goodness of CLMLS using linear regression
def conditional_log_marginal_likelihood_sequence(cov_func, X_seq, Y_seq, whole_sequence=False):
    seq_len = len(X_seq)
    clml_functions = []
    for i in range(seq_len-1):
        # X_con = [np.concatenate(X_seq[:i+1], axis=0), X_seq[seq_len-1]]
        # Y_con = [np.concatenate(Y_seq[:i+1], axis=0), Y_seq[seq_len-1]]
        X_con = [np.concatenate(X_seq[:i+1], axis=0), X_seq[i+1]]
        Y_con = [np.concatenate(Y_seq[:i+1], axis=0), Y_seq[i+1]]
        clml_function = conditional_log_marginal_likelihood(cov_func, X_con, Y_con)
        clml_functions.append(clml_function)
    #

    def clmls_function(unconstrained_hyperparams):
        clmls_vals = [f(unconstrained_hyperparams) for f in clml_functions]
        ab = opt_clmls(clmls_vals)
        # print(np.array(clmls_vals))
        # print(abcd)
        if whole_sequence:
            return np.array(clmls_vals)
        return ab[0] + ab[1] * len(clmls_vals) # calculate next value of sequence as prediction
    #

    return clmls_function
#

# optimization loop based on CLMLS
def clmls_opt(cov_func, X_seq, Y_seq, unconstrained_hyperparams_init, step_size, T):
    clmls_function = conditional_log_marginal_likelihood_sequence(cov_func, X_seq, Y_seq)
    val_grad_function = value_and_grad(clmls_function)
    unconstrained_hyperparams = unconstrained_hyperparams_init
    for i in range(T):
        val, grad = val_grad_function(unconstrained_hyperparams)
        print(i, val)
        unconstrained_hyperparams += grad * step_size
    #
    # print(val)
    return (unconstrained_hyperparams, val)
#


