# -*- coding: utf8 -*-
import torch
import numpy as np
import random
import pickle
from torch import nn, autograd
from torch.nn.utils import vector_to_parameters
from Library.general_functions import distribute_dataset_to_clients, random_sampling


class RBF(nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y, compute_partition=False, layer_names=None, layer_positions=None):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().clone().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-10 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        grad_K = autograd.grad(K_XY.sum(), X)[0]

        return K_XY, grad_K


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=True):
        super(MLP, self).__init__()
        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            if self.activation:
                self.hidden_layers.append(nn.ReLU())
            input_size = hidden_size

        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


def grad_log_kde(N, x, y, bandwidth=-1, kernel='gaussian'):
    xt, yt = x.transpose(0, 1), y.transpose(0, 1)
    x_squared = xt.norm(dim=0) ** 2
    new_x = x_squared.view(1, -1).repeat(N, 1).transpose(0, 1)
    y_squared = yt.norm(dim=0) ** 2
    new_y = y_squared.view(1, -1).repeat(N, 1)
    distances_squared = new_x + new_y - 2 * torch.mm(xt.transpose(0, 1), yt)

    back_weights = torch.ones(N, device=x.device)

    if kernel == 'gaussian':
        exp_distances = torch.exp(distances_squared * (-0.5) * (1 / bandwidth ** 2))
        sum_exp_distances = exp_distances.sum(dim=1)
    else:
        raise NotImplementedError

    log_q = torch.log(sum_exp_distances + 10 ** (-10))
    log_q.backward(back_weights)

    x_grad = x.grad.detach().clone()
    x.grad.zero_()

    return x_grad


def evaluation(M, model, theta, device, X_test, y_test, get_prob=None):
    """
    Returns accuracy and log-likelihood for particles based methods.
    Set get_pred \neq None to get confidence.
    """
    acc = np.zeros(M)
    llh = np.zeros(M)

    model.to(device)
    X_test = torch.from_numpy(X_test).to(device)
    '''
        Since we have M particles, we use a Bayesian view to calculate accuracy and log-likelihood
        where we average over the M particles
    '''
    with torch.no_grad():
        for i in range(M):
            vector_to_parameters(theta[i].detach().clone(), model.parameters())
            model.eval()
            outputs = model(X_test)
            prob, pred_y_test = torch.max(torch.softmax(outputs, dim=1), 1)
            prob = prob.cpu().detach().numpy()
            pred_y_test = pred_y_test.cpu().detach().numpy()

            acc[i] = len(np.where(pred_y_test == y_test)[0]) / len(y_test)
            llh[i] = np.mean(np.log(prob))
            model.train()

    if get_prob is None:
        return (np.mean(acc), np.mean(llh))
    else:
        # get predictive distribution (confidence) by averaging over particles
        return np.mean(pred_y_test, axis=0)


def initialize_model_params(model, mean=0, std=1):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)


def agent_dsvgd(i, M, model, device, nb_svgd, X_train, y_train, global_particles, local_particles, batchsize, lr):
    '''

    :param i:
    :param M:
    :param model:
    :param device:
    :param nb_svgd:
    :param X_train:
    :param y_train:
    :param global_particles: == global_poster
    :param local_particles: == global_likely
    :param batchsize:
    :param lr:
    :return:
    '''

    particles = global_particles.detach().clone()
    particles.requires_grad = True
    global_particles.requires_grad = False

    model.zero_grad()
    model.to(device)

    N0 = X_train.shape[0]
    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)

    grad_theta = torch.zeros_like(particles).to(device)

    svgd_kernel = RBF()

    optimizer = torch.optim.Adam([particles], lr=lr[0], betas=adam_betas, eps=adam_eps)
    criterion = nn.CrossEntropyLoss()

    for t in range(nb_svgd):
        kxy, dxkxy = svgd_kernel(particles, particles.detach())

        ''' qi_1 = q^(i-1) '''
        grad_logqi_1 = grad_log_kde(M, particles, global_particles, my_lambda)

        ''' t^(i-1) '''
        grad_log_ti_1 = grad_log_kde(M, particles, local_particles, my_lambda)

        ''' Compute grad_theta '''
        batch = [ii % N0 for ii in range(t * batchsize, (t + 1) * batchsize)]
        for m in range(M):
            theta = particles[m, :].detach().clone()
            vector_to_parameters(theta, model)
            model.train()

            outputs = model(X_train[batch, :])

            log_lik_data = -criterion(outputs, y_train[batch]) * N0
            log_lik_data.backward()

            gradients = torch.cat([param.grad.detach().clone().view(-1) for param in model.parameters()])
            grad_theta[m, :] = gradients.detach().clone()
            model.zero_grad()

        grad_sv_target = grad_theta.detach().clone().double() / alpha + grad_logqi_1 - grad_log_ti_1

        delta_theta = (1 / M) * (
                torch.mm(kxy.detach().clone(), grad_sv_target.detach().clone()) + dxkxy.detach().clone())

        particles.grad = -delta_theta.detach().clone()
        optimizer.step()
        optimizer.zero_grad()

    return particles.detach().clone()


def server(nb_devices, model, particles, M, device, nb_svgd_1, nb_svgd_2, nb_svgd_3, nb_global, client_X_train,
           client_y_train, X_test, y_test, batchsize, lr):
    client_poster = particles.repeat(nb_devices, 1, 1)
    global_poster = particles.detach().clone()
    global_likely = particles.detach().clone()

    acc = np.zeros(nb_global)
    llh = np.zeros(nb_global)

    svgd_kernel = RBF()

    for i in range(0, nb_global):

        client_subset = random_sampling(nb_devices, client_sample_ratio)

        for idx, curr_id in enumerate(client_subset):
            X_curr = client_X_train[curr_id]
            y_curr = client_y_train[curr_id]

            client_poster[curr_id], t_iteration_count, t_accuracy, t_loss = \
                agent_dsvgd(i, M, model, device, nb_svgd_1, X_curr, y_curr, global_poster.detach().clone(),
                            global_likely.detach().clone(), batchsize, lr)

            tacc, tllh = evaluation(M, model, client_poster[curr_id].detach().clone(), device, X_test, y_test)
            print("round {} : agent {} : acc ==> {}, llh ==> {}".format(i, curr_id, tacc, tllh))

        global_poster_new = global_poster.detach().clone()
        global_poster_new.requires_grad = True
        global_poster.requires_grad = False

        optimizer = torch.optim.Adam([global_poster_new], lr=lr[1], betas=adam_betas, eps=adam_eps)

        """ SVGD updates the global posterior """
        for t in range(nb_svgd_2):
            kxy, dxkxy = svgd_kernel(global_poster_new, global_poster_new.detach())

            ''' q^(i-1) '''
            grad_log_q_old = grad_log_kde(M, global_poster_new, global_poster, my_lambda)

            ''' q_k '''
            grad_log_q_k_sum = torch.zeros_like(global_poster_new)
            for curr_id in client_subset:
                grad_log_q_k_sum += grad_log_kde(M, global_poster_new, client_poster[curr_id], my_lambda)

            grad_log_qi = grad_log_q_old * (1 - len(client_subset)) + grad_log_q_k_sum

            delta_theta = (1 / M) * (
                    torch.mm(kxy.detach().clone(), grad_log_qi.detach().clone()) + dxkxy.detach().clone())

            global_poster_new.grad = -delta_theta.detach().clone()

            optimizer.step()
            optimizer.zero_grad()

        global_poster_new.requires_grad = False
        global_likely_new = global_likely.detach().clone()
        global_likely_new.requires_grad = True
        global_likely.requires_grad = False

        optimizer = torch.optim.Adam([global_likely_new], lr=lr[2], betas=adam_betas, eps=adam_eps)

        ''' SVGD updates the average likelihood '''
        for t in range(nb_svgd_3):
            kxy, dxkxy = svgd_kernel(global_likely_new, global_likely_new.detach())

            ''' t^(i-1) '''
            grad_log_t_old = grad_log_kde(M, global_likely_new, global_likely, my_lambda)

            ''' q^i '''
            grad_log_q_new = grad_log_kde(M, global_likely_new, global_poster_new, my_lambda)

            ''' q^(i-1) '''
            grad_log_q_old = grad_log_kde(M, global_likely_new, global_poster, my_lambda)

            grad_log_t_new = (grad_log_q_new - grad_log_q_old) / (nb_devices * 1.0) + grad_log_t_old

            delta_theta = (1 / M) * (
                    torch.mm(kxy.detach().clone(), grad_log_t_new.detach().clone()) + dxkxy.detach().clone())

            global_likely_new.grad = -delta_theta.detach().clone()

            optimizer.step()
            optimizer.zero_grad()

        global_poster = global_poster_new.detach().clone()
        global_likely = global_likely_new.detach().clone()

        acc[i], llh[i] = evaluation(M, model, global_poster.detach().clone(), device, X_test, y_test)
        print("round {} :  acc ==> {}, llh ==> {}\n".format(i, acc[i], llh[i]))

    return acc, llh


if __name__ == '__main__':
    ''' Parameters '''
    torch.random.manual_seed(0)
    np.random.seed(0)
    alpha = 1.  # Temperature parameters
    nb_svgd_1 = 200  # Iteration number of the first SVGD loop, updating the client-side model
    nb_svgd_2 = 200  # The number of iterations of the second SVGD loop, updating the client likelihood
    nb_svgd_3 = 200  # The number of iterations of the third SVGD loop, updating the global posterior particles
    nb_global = 50  # Indicates the number of iterations of PVI (Particle Variational Inference)
    my_lambda = -1
    number_labels = 10  # The number of label types in the MNIST dataset
    nb_exp = 10  # The number of random trials. This value determines the number of trials to average the results.
    nb_devices = 20  # Number of agents (devices)
    client_sample_ratio = 0.1  # Client sampling ratio
    train_ratio = 0.8  # The proportion of training data set
    batchsize = 128
    loglambda = 1  # Logarithmic precision of weight priors
    hidden_size = [100]  # The number of neurons in the hidden layer
    non_iid = True  # Data distribution
    dirichlet_alpha = 1.0  # Dirichlet distribution parameter alpha
    M = 20  # number of particles
    adam_betas = (0.9, 0.99)
    adam_eps = 1e-8
    lr = [0.01, 0.01, 0.01]  # [local_lr, global_lr_1, global_lr_2]
    array_acc = np.zeros(nb_global)
    array_llh = np.zeros(nb_global)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''' load data file '''
    with open("./data/pickled_mnist.pkl", "br") as fh:
        data = pickle.load(fh)

    train_imgs = data[0]
    test_imgs = data[1]
    train_labels = data[2].astype(int)
    test_labels = data[3].astype(int)

    for exp in range(nb_exp):
        ''' build the training and testing data set '''
        permutation = np.arange(train_imgs.shape[0])
        random.shuffle(permutation)
        index_train = permutation

        permutation = np.arange(test_imgs.shape[0])
        random.shuffle(permutation)
        index_test = permutation

        X_train, y_train = train_imgs[index_train, :], train_labels[index_train].flatten()
        X_test, y_test = test_imgs[index_test], test_labels[index_test].flatten()

        ''' divide the dataset '''
        if non_iid:
            client_X_train, client_y_train = distribute_dataset_to_clients(X=X_train, y=y_train, n_clients=nb_devices,
                                                                           alpha=dirichlet_alpha, show=True)
            for i in range(nb_devices):
                permutation = np.arange(client_X_train[i].shape[0])
                random.shuffle(permutation)
                index_train = permutation
                client_X_train[i], client_y_train[i] = client_X_train[i][index_train, :], client_y_train[i][index_train]
        else:
            client_X_train = []
            client_y_train = []
            for curr_id in range(nb_devices):
                X_curr = X_train[
                         curr_id * X_train.shape[0] // nb_devices: ((curr_id + 1) * X_train.shape[0]) // nb_devices, :]
                y_curr = y_train[
                         curr_id * X_train.shape[0] // nb_devices: ((curr_id + 1) * X_train.shape[0]) // nb_devices]
                client_X_train.append(X_curr)
                client_y_train.append(y_curr)

        ''' model '''
        input_size = X_train.shape[1]
        output_size = number_labels
        model = MLP(input_size=input_size, hidden_sizes=hidden_size, output_size=output_size)
        model.to(device)
        model.to(torch.double)
        num_vars = sum(p.numel() for p in model.parameters())

        ''' Initialize particles'''
        particles = torch.zeros(size=(M, num_vars), dtype=torch.double).to(device)
        for m in range(M):
            initialize_model_params(model=model)
            temp = torch.cat([param.detach().clone().view(-1) for param in model.parameters()])
            particles[m, :] = temp.detach().clone()

        ''' Run DSVGD server with Round Robin (RR) scheduling '''
        curr_acc, cur_llh = server(nb_devices, model, particles, M, device, nb_svgd_1, nb_svgd_2, nb_svgd_3, nb_global,
                                   client_X_train, client_y_train, X_test, y_test, batchsize, lr)
        array_acc += curr_acc
        array_llh += cur_llh

    print('BNN multilabel classification accuracy with sDSVGD as function of comm. rounds = {}'.format(
        repr(array_acc / nb_exp)))
    print('BNN multilabel classification llh with sDSVGD as function of comm. rounds = {}'.format(
        repr(array_llh / nb_exp)))
