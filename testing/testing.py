import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ctm_dataloader import create_dataloader
import pandas as pd
import numpy as np

class CTM(nn.Module):
    def __init__(self, num_topics, vocab_size, rho_size):
        super(CTM, self).__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.rho_size = rho_size

        # Parameters for document-topic distribution
        self.alpha = nn.Parameter(torch.randn(num_topics))

        # Parameters for topic-word distribution
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))

        # Fixed parameters for the Gaussian distribution of the correlation matrix
        self.mu = nn.Parameter(torch.zeros(num_topics))
        self.sigma = nn.Parameter(torch.ones(num_topics, num_topics))

    def forward(self, bow):
        # Calculate document-topic distribution using the softmax function
        theta = F.softmax(self.alpha, dim=0)

        # Calculate topic-word distribution using the softmax function
        phi = F.softmax(self.beta, dim=1)

        # Sample correlation matrix from a Gaussian distribution
        rho = torch.randn_like(self.mu) * self.sigma + self.mu
        sigma = torch.mm(rho, rho.t())

        # Calculate the document-topic distribution for each document in the batch
        # doc_topic_dist = torch.mm(torch.mm(bow, phi.t()), torch.diag(theta))
        doc_topic_dist = torch.mm(bow, torch.mm(theta.diag(), phi).t()).long()

        return doc_topic_dist, theta, phi, sigma

    def ctm_loss(self, doc_topic_dist, bow, theta, phi, sigma):
        # Reconstruction loss
        recon_loss = -torch.sum(bow * torch.log(doc_topic_dist + 1e-9))

        # Regularization terms
        alpha_reg = -0.5 * torch.sum(theta * theta)
        beta_reg = -0.5 * torch.sum(phi * phi)
        rho_reg = -0.5 * torch.sum(sigma * sigma)

        # Total loss
        total_loss = recon_loss + alpha_reg + beta_reg + rho_reg

        return total_loss

    def train_ctm(self, model, dataloader, optimizer, num_epochs):
        model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0

            for idx, bow in enumerate(dataloader):
                optimizer.zero_grad()

                # Forward pass
                doc_topic_dist, theta, phi, sigma = model(bow)

                # Calculate loss
                loss = self.ctm_loss(doc_topic_dist, bow, theta, phi, sigma)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


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
    num_topics = 5
    batch_size = 64
    rho_size = 5
    lr = 0.001

    # Create dataloader
    train_loader, vocab_size = create_dataloader(documents, batch_size)
    
    for idx, data in enumerate(train_loader):
        print("Shape of data:", data.shape)
        break

    model = CTM(num_topics, vocab_size, rho_size)

    # Hyper parameters for the model
    optimizer = optim.Adam(model.parameters(), lr = lr)

    # Model training
    model.train_ctm(model, train_loader, optimizer, num_epochs)
