import math
import torch
from GSSFiltering.filtering import Extended_Kalman_Filter, KalmanNet_Filter, KalmanNet_Filter_v2, Split_KalmanNet_Filter, Cholesky_KalmanNet_Filter
import time
from datetime import timedelta

print_num = 25

def mse(target, predicted):
    """Mean Squared Error"""
    return torch.mean(torch.square(target - predicted))
    # return torch.sum(torch.square(target - predicted))

def mse_psd(target, predicted_mean, predicted_var, LD=0):
    """Mean Squared Error + Positive Semi definite constraints"""
    L1 = mse(target, predicted_mean)
    eig_vals = torch.linalg.eigvals(predicted_var).real
    penalty = torch.sum(torch.clamp(-eig_vals, min=0))
    # eig_vals_clamped = torch.transpose(eig_vals_clamped, 1, 2)
    L2 = LD * penalty
    return L1 + L2, L1, L2

def empirical_averaging(target, predicted_mean, predicted_var, beta=0.5):
    L1 = mse(target, predicted_mean)
    # L2 = torch.sum(torch.abs((target - predicted_mean)**2 - predicted_var))
    L2 = torch.mean(torch.abs((target - predicted_mean)**2 - predicted_var))

    return (1-beta)*L1 + beta * L2, L1, L2

def EA_psd(target, predicted_mean, predicted_var, beta, LD=0.2):
    """Mean Squared Error + Positive Semi definite constraints"""
    L1 = empirical_averaging(target, predicted_mean, predicted_var, beta)
    eig_vals = torch.linalg.eigvals(predicted_var).real
    penalty = torch.sum(torch.clamp(-eig_vals, min=0))
    # eig_vals_clamped = torch.transpose(eig_vals_clamped, 1, 2)
    L2 = LD * penalty
    return L1 + L2, L1, L2

def gaussian_nll(target, predicted_mean, predicted_var):
    """Gaussian Negative Log Likelihood (assuming diagonal covariance)"""
    # predicted_var += 1e-12
    mahal = torch.square(target - predicted_mean) / torch.abs(predicted_var)
    element_wise_nll = 0.5 * (torch.log(torch.abs(predicted_var)) + torch.log(torch.tensor(2 * torch.pi)) + mahal)
    sample_wise_error = torch.sum(element_wise_nll, dim=-2)
    return torch.mean(sample_wise_error)

class Tester():
    def __init__(self, filter, data_path, model_path, is_validation=False, is_mismatch=False):
        # Example:
        #   data_path = './.data/syntheticNL/test/(true)
        #   model_path = './.model_saved/(syntheticNL) Split_KalmanNet_5000.pt'

        if isinstance(filter, Extended_Kalman_Filter):
            self.result_path = 'EKF '
        if isinstance(filter, KalmanNet_Filter):
            self.result_path = 'KN v1 '
        if isinstance(filter, KalmanNet_Filter_v2):
            self.result_path = 'KN v2 '
        if isinstance(filter, Split_KalmanNet_Filter):
            self.result_path = 'SKN '
        if isinstance(filter, Cholesky_KalmanNet_Filter):
            self.result_path = 'CKN '


        self.filter = filter
        if not isinstance(filter, Extended_Kalman_Filter):
            self.filter.kf_net = torch.load(model_path)
            self.filter.kf_net.initialize_hidden()
        self.x_dim = self.filter.x_dim
        self.y_dim = self.filter.y_dim
        self.data_path = data_path
        self.model_path = model_path
        self.is_validation = is_validation
        self.is_mismatch = is_mismatch

        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        self.data_x = torch.load(data_path + 'state.pt')
        self.data_y = torch.load(data_path + 'obs.pt')
        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]
        assert(self.x_dim == self.data_x.shape[1])
        assert(self.y_dim == self.data_y.shape[1])
        assert(self.seq_len == self.data_y.shape[2])
        assert(self.data_num == self.data_y.shape[0])

        x_hat = torch.zeros_like(self.data_x)
        cov_hat = torch.zeros(self.data_num, self.seq_len, self.x_dim, self.x_dim)
        cov_hat_diag = torch.zeros(self.data_num, self.x_dim, self.seq_len) 

        if isinstance(filter, Split_KalmanNet_Filter) or isinstance(filter, Cholesky_KalmanNet_Filter) :
            Pk_hat = torch.zeros(self.data_num, self.seq_len, self.x_dim, self.x_dim)    
            Sk_hat = torch.zeros(self.data_num, self.seq_len, self.y_dim, self.y_dim)
         
        start_time = time.monotonic()

        with torch.no_grad():
            for i in range(self.data_num):
                if i % print_num == 0:
                    if self.is_validation:
                        print(f'Validating {i} / {self.data_num} of {self.model_path}')
                    else:
                        print(f'Testing {i} / {self.data_num} of {self.model_path}')
                
                self.filter.state_post = self.data_x[i,:,0].reshape((-1,1))

                for ii in range(1, self.seq_len):
                    self.filter.filtering(self.data_y[i,:,ii].reshape((-1,1)))
                x_hat[i] = self.filter.state_history[:,-self.seq_len:]  
                cov_hat[i] = self.filter.cov_history[-self.seq_len:, :, :]
                cov_hat_diag[i] = (torch.diagonal(cov_hat[i], dim1=-2, dim2=-1)).T

                if isinstance(filter, Split_KalmanNet_Filter) or isinstance(filter, Cholesky_KalmanNet_Filter) :
                    Pk_hat[i] = self.filter.Pk_history[-self.seq_len:, :, :]
                    Sk_hat[i] = self.filter.Sk_history[-self.seq_len:, :, :]

                self.filter.reset(clean_history=False)

            end_time = time.monotonic()
            # print(timedelta(seconds=end_time - start_time))

            torch.save(x_hat, data_path + self.result_path + 'x_hat.pt')
            torch.save(cov_hat, data_path + self.result_path + 'cov_hat.pt')

            if isinstance(filter, Split_KalmanNet_Filter) or isinstance(filter, Cholesky_KalmanNet_Filter) :
                torch.save(Pk_hat, data_path + self.result_path + 'Pk_hat.pt')
                torch.save(Sk_hat, data_path + self.result_path + 'Sk_hat.pt')

            # loss function ===========================================================
            loss = self.loss_fn(self.data_x[:,:,1:], x_hat[:,:,1:])
            # =========================================================================

            print(f'loss = {loss:.4f}')

            # Compute loss at instantaneous time
            self.loss_instant = torch.zeros(self.data_x[:,:,1:].shape[-1])
            for i in range(self.data_x[:,:,1:].shape[-1]):
                self.loss_instant[i] = self.loss_fn(self.data_x[:,:,i+1], x_hat[:,:,i+1])
            self.loss_instant_dB = 10*torch.log10(self.loss_instant)

        self.loss = loss
        self.x_hat = x_hat
        self.cov_hat = cov_hat
        self.cov_hat_diag = cov_hat_diag
