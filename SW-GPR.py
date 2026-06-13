from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import cma
from sklearn.preprocessing import MinMaxScaler
import re

def getdata(file_path):
    all_data = pd.DataFrame()
    all_data = pd.read_excel(file_path)
    all_data.dropna(inplace=True)
    all_data = all_data.values
    y = all_data[:, :1]
    X = all_data[:, 1:]
    y= y.ravel()

    return X, y

class GPR:

    def __init__(self, optimize=True):
        '''
        train_X, train_y: Training set
        params: Kernel function parameters, including l, sigma_f, noise level of source domain sigma_s, noise level of target domain sigma_t
        lambda_matrix: Diagonal weight matrix
        n_source, n_target: Number of samples from source domain and target domain in the training set
        Kyy: kernel(train_X, train_X)
        '''
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 1, "sigma_f": 1, "sigma_s": 0.1, "sigma_t": 0.1}
        self.optimize = optimize
        self.Kyy = None
        self.weight = None
        self.lambda_matrix = None
        self.n_source = None
        self.n_target = None
        self.half_result = None
        self.loss = None
    
    def fit(self, X, y, n_source, n_target, limit):
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        self.n_source = n_source
        self.n_target = n_target
        self.weight = np.ones(n_source) * limit
        self.weightlimit = limit


        def negative_log_likelihood_loss(params):
            
            self.lambda_matrix = np.diag(np.hstack((self.weight, np.ones(n_target))))
            self.params["l"], self.params["sigma_f"], self.params["sigma_s"], self.params["sigma_t"] = params[0], params[1], params[2], params[3]
            Kyy = self.kernel(self.train_X, self.train_X)
            #K' = λKλ - diag(λKλ) + diag(K)
            lambda_K_lambda = self.lambda_matrix @ Kyy @ self.lambda_matrix  # λKλ
            diag_lambda_K_lambda = np.diag(np.diagonal(lambda_K_lambda))  # diag(λKλ)
            diag_K = np.diag(np.diagonal(Kyy))  # diag(K)  
            A = np.diag(np.hstack((np.full(n_source, self.params["sigma_s"] ** 2), np.full(n_target, self.params["sigma_t"] ** 2))))
            Kyy = lambda_K_lambda - diag_lambda_K_lambda + diag_K + A

            self.Kyy = Kyy
            L = np.linalg.cholesky(Kyy + 1e-8 * np.eye(len(self.train_X)))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.train_y))
            loss = 0.5 * self.train_y.T.dot(alpha) + np.sum(np.log(np.diag(L))) \
            + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            self.loss = loss
            
            return loss

        if self.optimize:
            x0 = np.hstack(( [self.params["l"], self.params["sigma_f"], self.params["sigma_s"], self.params["sigma_t"]], self.weight ))
            sigma0 = 0.5
            bounds_lower = np.hstack(( [np.log(1e-4), np.log(1e-4), 0.05, 0.05], [0]*len(self.weight) ))
            bounds_upper = np.hstack(( [np.log(1e4), np.log(1e4), 1, 1], [self.weightlimit]*len(self.weight) ))
            opts = {'bounds': [bounds_lower, bounds_upper],
                    'verb_disp': 1,
                    'maxiter': 500
                    }
            res = cma.fmin(negative_log_likelihood_loss, x0, sigma0, options=opts, restarts=0)
            best_params = res[0]
            self.params["l"] = np.exp(best_params[0])
            self.params["sigma_f"] = np.exp(best_params[1])
            self.params["sigma_s"] = best_params[2]
            self.params["sigma_t"] = best_params[3]
            self.weight = best_params[4:]

        self.is_fit = True


    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        N = len(self.train_X)
        jitter = 1e-8

        Kff = self.kernel(self.train_X, self.train_X) + jitter * np.eye(N)
        Kyy = self.kernel(X, X)
        Kfy = self.kernel(self.train_X, X)

        L = np.linalg.cholesky(Kff) 
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.train_y)) 

        mu = Kfy.T @ alpha

        v = np.linalg.solve(L, Kfy)
        cov = Kyy - v.T @ v + self.params["sigma_t"] ** 2 * np.eye(len(X))

        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)


mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Sampling standard deviation")

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path1 = os.path.join(base_dir, 'feature', 'nasa', 'B0005.xlsx')
file_path2 = os.path.join(base_dir, 'feature', 'nasa', 'B0006.xlsx')
file_path3 = os.path.join(base_dir, 'feature', 'nasa', 'B0007.xlsx')
file_path4 = os.path.join(base_dir, 'feature', 'nasa', 'B0034.xlsx')

X1, y1 = getdata(file_path1)
X2, y2 = getdata(file_path3)
X3, y3 = getdata(file_path4)
Xt, yt = getdata(file_path2)

Xs = X1
ys = y1
weight_limit = 1
label = 'B0005->B0006'
scalerXs = MinMaxScaler(feature_range=(0, 1))
Xs = scalerXs.fit_transform(Xs)
scalerys = MinMaxScaler(feature_range=(0, 1))
ys = ys.reshape(-1, 1) 
ys = scalerys.fit_transform(ys)
scalerXt = MinMaxScaler(feature_range=(0, 1))
Xt = scalerXt.fit_transform(Xt)
scaleryt = MinMaxScaler(feature_range=(0, 1))
yt = yt.reshape(-1, 1) 
yt = scaleryt.fit_transform(yt)
ys = ys.reshape(-1)
yt = yt.reshape(-1)


test_size = 0.9
Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, test_size=test_size, random_state=42)

X_train2 = np.vstack((Xs, Xt_train))
y_train2 = np.hstack((ys, yt_train))
X_test = Xt_test
y_test = yt_test

n_source = int(ys.shape[0])
n_target_train = int(yt_train.shape[0])
n_target_all = int(yt.shape[0])

np.set_printoptions(suppress=True, precision=4)

sw_gpr = GPR()
sw_gpr.fit(X_train2, y_train2, n_source, n_target_train, weight_limit)

best_weight = sw_gpr.weight
optimize_K = sw_gpr.Kyy
swpgr_test_pred, cov_test = sw_gpr.predict(X_test)
swpgr_yt_pred, cov_test = sw_gpr.predict(Xt)

yt = yt.reshape(-1, 1) 
yt = scaleryt.inverse_transform(yt)
yt = yt.reshape(-1)
ys = ys.reshape(-1, 1) 
ys = scalerys.inverse_transform(ys)
ys = ys.reshape(-1)
swpgr_yt_pred = swpgr_yt_pred.reshape(-1, 1) 
swpgr_yt_pred = scaleryt.inverse_transform(swpgr_yt_pred)
swpgr_yt_pred = swpgr_yt_pred.reshape(-1)

swgpr_RMSE = np.sqrt(mean_squared_error(yt,swpgr_yt_pred))
swgpr_MAE = mean_absolute_error(yt,swpgr_yt_pred)
print(f"SW-GPR RMSE:{round(swgpr_RMSE, 4)}")
print(f"SW-GPR MAE:{round(swgpr_MAE, 4)}")

window_size = 5
def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')
pred_smoothed = moving_average(swpgr_yt_pred, window_size)
smooth_variance = np.var(pred_smoothed)
error_smoothed = moving_average(swpgr_yt_pred - yt, window_size)
smooth_error_variance = np.var(error_smoothed)
print(f"SW-GPR SV: {smooth_variance:.6f}")

z_score = 1.96
cov_diag = np.maximum(np.diag(cov_test), 0.0)
std_norm = np.sqrt(cov_diag)
std_original = std_norm / scaleryt.scale_[0]
pred_upper = swpgr_yt_pred + z_score * std_original
pred_lower = swpgr_yt_pred - z_score * std_original
R = np.max(yt) - np.min(yt)
interval_widths = pred_upper - pred_lower
NMPIW = np.mean(interval_widths / R)
print(f"NMPIW: {NMPIW:.6f}")


plt.figure(figsize=(12, 6))
plt.plot(yt, label='True SOH', color='#1f77b4', linewidth=2)
plt.plot(swpgr_yt_pred, label=f'{label} GPR Predicted SOH', color='#ff7f0e', linestyle='--', linewidth=2)
plt.fill_between(range(len(yt)), pred_lower, pred_upper, 
                 color='#ff7f0e', alpha=0.2, label='95% Confidence Interval')
plt.xlabel('Cycle Number', fontsize=12)
plt.ylabel('SOH', fontsize=12)
plt.title(f'{label} GPR Battery SOH Prediction with 95% Confidence Interval', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()

plt.imshow(optimize_K, cmap='coolwarm')
plt.colorbar(label='Value')
plt.title('Matrix Visualization (High values in dark color)')
plt.show()
