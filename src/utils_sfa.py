import numpy as np

def matrix_Fourier(N, w):
    t = np.arange(N).reshape((1, N))
    k = np.arange(w).reshape((w, 1))+1
    E = np.cos(2*np.pi * k * t / N)-1j*np.sin(2*np.pi * k * t / N)
    return E


def compute_Fourier(Y, w):
    (N1, N2) = np.shape(Y)
    Z = np.zeros((w, N1), dtype=complex)
    E = matrix_Fourier(N2, w)
    for i in range(N1):
        Z[:, i] = np.matmul(E, Y[i, :])
    return Z


def compute_symbolization_train(Z, A):
    (w, N1) = np.shape(Z)
    Z_num = np.zeros((w, N1), dtype=complex)
    Q_num = np.zeros((w, A-1), dtype=complex)
    for i in range(w):
        q_real = np.percentile(np.real(Z[i, :]), 100*(np.arange(A-1)+1)/A)
        out_real = np.searchsorted(q_real, np.real(Z[i, :]))
        q_imag = np.percentile(np.imag(Z[i, :]), 100*(np.arange(A-1)+1)/A)
        out_imag = np.searchsorted(q_imag, np.imag(Z[i, :]))
        Z_num[i, :] = out_real+1j*out_imag
        Q_num[i, :] = q_real+1j*q_imag
    return Z_num, Q_num


def compute_symbolization_test(Z, A, quantile):
    (w, N1) = np.shape(Z)
    Z_num = np.zeros((w, N1), dtype=complex)
    for i in range(w):
        q_real = np.real(quantile[i, :])
        out_real = np.searchsorted(q_real, np.real(Z[i, :]))
        q_imag = np.imag(quantile[i, :])
        out_imag = np.searchsorted(q_imag, np.imag(Z[i, :]))
        Z_num[i, :] = out_real+1j*out_imag
    return Z_num


def compute_means(Z, Z_num, A):
    (w, N1) = np.shape(Z)
    M = np.zeros((w, A), dtype=complex)
    for i in range(w):
        for j in range(A):
            z_real = np.real(Z_num[i, :])
            z_imag = np.imag(Z_num[i, :])
            M[i, j] = np.mean(np.real(Z[i, z_real == j])) + \
                1j*np.mean(np.imag(Z[i, z_imag == j]))
    return M


def compute_reconstruction(Z_num, M):
    (w, N1) = np.shape(Z_num)
    Z_recon = np.zeros((w, N1), dtype=complex)
    for i in range(w):
        z_real = np.real(Z_num[i, :]).astype(int)
        z_imag = np.imag(Z_num[i, :]).astype(int)
        Z_recon[i, :] = np.real(M[i, z_real])+1j*np.imag(M[i, z_imag])
    return Z_recon


def inverse_Fourier(Z_recon, Y, w):
    (N1, N2) = np.shape(Y)
    E = matrix_Fourier(N2, w)
    Y_hat = np.zeros((N1, N2))
    for i in range(N1):
        Y_hat[i, :] = np.real(2*np.matmul(E.conj().T, Z_recon[:, i])/N2)
    return Y_hat


def compute_SFA_reconstruct(Y, w, A):
    Z = compute_Fourier(Y, w)
    Z_num = compute_symbolization(Z, A)
    M = compute_means(Z, Z_num, A)
    Z_recon = compute_reconstruction(Z_num, M)
    Y_hat = inverse_Fourier(Z_recon, Y, w)
    return Y_hat


def train_SFA(Y, w, A):
    Z = compute_Fourier(Y, w)
    Z_num, quantile = compute_symbolization_train(Z, A)
    M = compute_means(Z, Z_num, A)
    return M, quantile


def train_SFA(Y, w, A):
    Z = compute_Fourier(Y, w)
    Z_num, quantile = compute_symbolization_train(Z, A)
    M = compute_means(Z, Z_num, A)
    return M, quantile


def test_SFA(Y, w, A, M, quantile):
    Z = compute_Fourier(Y, w)
    Z_num = compute_symbolization_test(Z, A, quantile)
    Z_recon = compute_reconstruction(Z_num, M)
    Y_hat = inverse_Fourier(Z_recon, Y, w)
    return Y_hat


def compute_RMSE(Y, Y_hat):
    (N1, N2) = np.shape(Y)
    rmse = np.zeros((N1,))
    for i in range(N1):
        rmse[i] = np.sqrt(np.mean(np.abs(Y_hat[i, :]-Y[i, :])**2))
    return rmse
