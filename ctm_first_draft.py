import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ctm_dataloader import create_dataloader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.wishart import Wishart
from torch.nn.functional import softmax


class CTM(nn.Module):
    def __init__(self, num_topics, vocab_size, rho):
        super(CTM, self).__init__()

        # num_topics: number of topics in the model
        # vocab_size: size of the vocabulary
        # rho: hyperparameter for the model
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.rho = rho
        self.alpha = nn.Parameter(torch.ones(num_topics))
    
        # beta: topic-word distribution
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))

        # covariance matrix for z
        self.sigma = nn.Parameter(torch.zeros(num_topics, num_topics) + 1e-6 * torch.eye(num_topics))

        # mu: topic bias
        self.mu = nn.Parameter(torch.randn(num_topics))

    # def forward(self, x):
    #     # x is a batch of documents represented as a bag-of-words
    #     # x has shape (batch_size, vocab_size)
    #     batch_size = x.size(0)
    #     print(f"Batch size: {batch_size}")

    #     # log_prob: log probability of each document belonging to each topic
    #     log_prob = torch.zeros(batch_size, self.num_topics)
    #     print(f"Log probability {log_prob} and shape: {log_prob.shape}")
    #     for k in range(self.num_topics):

    #         # compute the log probability of each document belonging to topic k
    #         log_prob[:, k] = torch.sum(x * self.beta[k], dim=1) + self.mu[k]

    #         for j in range(self.num_topics):
    #             if j != k:
    #                 # compute contribution of topic j to the log probability
    #                 # of each document belonging to topic k
    #                 log_prob[:, k] += (
    #                     self.rho * self.theta[k, j] * torch.sum(x * self.beta[j], dim=1)
    #                 )
            
    #         print(f'Log probability for topic {k} is {log_prob[:, k]}')

    #     return log_prob
    
    def forward(self, bow):
        theta = softmax(self.alpha, dim=0)
        # print(f'Forward pass theta: {theta}')
        beta = softmax(self.beta, dim=1)
        # print(f'Forward pass beta: {beta}')
        z = torch.randn(bow.shape[0], self.num_topics)
        # print(f'Forward pass z before update: {z}')
        for i in range(bow.shape[0]):
            z[i] = MultivariateNormal(self.mu, self.sigma).sample()
            if i % 1000 == 0:
                print(f'Forward pass z at iteration {i}: {z[i]}')
        print(f'Forward pass z: {z}')
        eta = torch.exp(torch.matmul(z, beta))
        eta = eta / torch.sum(eta, dim=1, keepdim=True)
        gamma = torch.matmul(eta, torch.diag(theta))
        gamma = gamma + self.rho
        gamma = gamma / torch.sum(gamma, dim=1, keepdim=True)
        return gamma
    
    def loss_function(gamma, beta, z, mu, sigma):
        print(f'Loss function z: {z}')
        log_likelihood = torch.sum(torch.log(torch.matmul(gamma, beta)))
        log_prior = torch.sum(Dirichlet(torch.ones(beta.shape[0])).log_prob(beta))
        log_prior += torch.sum(MultivariateNormal(torch.zeros(mu.shape[0]), torch.eye(mu.shape[0])).log_prob(z))
        log_prior += torch.sum(MultivariateNormal(torch.zeros(sigma.shape[0]), torch.eye(sigma.shape[0])).log_prob(mu))
        log_prior += torch.sum(Wishart(torch.eye(sigma.shape[0]), sigma.shape[0]).log_prob(sigma))
        return -(log_likelihood + log_prior)

    def train_model(self, model, num_epochs, train_loader, optimizer):
        # Train loop
        for epoch in range(num_epochs):
            for i, data in enumerate(train_loader):
                # x = Variable(x.float())
                optimizer.zero_grad()
                print("Optimizing for iteration:", i + 1)
                print("Shape of data:", data[0].shape)
                output = model(data[0])
                print("Shape of output:", output.shape)
                # loss = criterion(output)
                loss = self.loss_function(output, self.beta, self.mu, self.sigma)
                print('Loss calculated', loss)
                # print(f"Loss: {loss} for iteration {i+1}")
                loss.backward()
                print("Backward pass done")
                optimizer.step()
                print("Optimizer step done")
                if (i + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}],\
                        Step [{i + 1}/{len(train_loader)}],\
                        Loss: {loss.item():.4f}"
                    )
                    
                break


if __name__ == "__main__":
    # Load the dataset
    news_data = pd.read_csv("newsgroups_data.csv")
    print("Shape of dataset:", news_data.shape)

    news_data = news_data.drop(columns=["Unnamed: 0"])

    documents = news_data.content
    target_labels = news_data.target
    target_names = news_data.target_names

    # Hyperparameters
    num_epochs = 10
    num_topics = target_names.shape[0]
    batch_size = 32
    rho = 0.1
    lr = 0.001

    # Create dataloader
    train_loader, vocab_size = create_dataloader(documents, batch_size)

    for idx, data in enumerate(train_loader):
        print("Shape of data:", data.shape)
        break

    model = CTM(num_topics, vocab_size, rho)

    # Hyper parameters for the model
    optimizer = optim.Adam(model.parameters(), lr = lr)
    # criterion = nn.CrossEntropyLoss()

    # Model training
    model.train_model(model, num_epochs, train_loader, optimizer)
