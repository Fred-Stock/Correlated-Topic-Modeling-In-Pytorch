import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from ctm_dataloader import create_dataloader

class CTM(nn.Module):
    def __init__(self, num_topics, vocab_size, rho):
        super(CTM, self).__init__()
        
        # num_topics: number of topics in the model
        # vocab_size: size of the vocabulary
        # rho: hyperparameter for the model
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.rho = rho
        
        # beta: topic-word distribution
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))
        
        # theta: topic-topic distribution
        self.theta = nn.Parameter(torch.randn(num_topics, num_topics))
        
        # mu: topic bias
        self.mu = nn.Parameter(torch.randn(num_topics))

    def forward(self, x):
        
        # x is a batch of documents represented as a bag-of-words
        # x has shape (batch_size, vocab_size)
        batch_size = x.size(0)
        print(f'Batch size: {batch_size}')
        
        # log_prob: log probability of each document belonging to each topic
        log_prob = torch.zeros(batch_size, self.num_topics)
        print(f'Log probability {log_prob} and shape: {log_prob.shape}')
        for k in range(self.num_topics):
            print(f'Topic {k}')
            
            # compute the log probability of each document belonging to topic k
            log_prob[:, k] = torch.sum(x * self.beta[k], dim=1) + self.mu[k]
            
            for j in range(self.num_topics):
                if j != k:
                    
                    # compute the contribution of topic j to the log probability of each document belonging to topic k
                    log_prob[:, k] += self.rho * self.theta[k, j] * torch.sum(x * self.beta[j], dim=1)
        
        return log_prob
    
def train_model(model, num_epochs, train_loader, optimizer, criterion):
    # Train loop
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            # x = Variable(x.float())
            optimizer.zero_grad()
            print('Optimizing for iteration:', i+1)
            output = model(data)
            print('Shape of output:', output.shape)
            loss = criterion(output)
            print(f'Loss: {loss} for iteration {i+1}')
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))


if __name__ == '__main__':
    
    # Load the dataset
    news_data = pd.read_csv('newsgroups_data.csv')
    print('Shape of dataset:', news_data.shape)
    
    news_data = news_data.drop(columns=['Unnamed: 0'])
    
    documents = news_data.content
    target_labels = news_data.target
    target_names = news_data.target_names
    
    # Hyperparameters
    num_epochs = 10
    num_topics = target_names.shape[0]
    batch_size = 32
    rho = 0.1
    
    # Create dataloader
    train_loader, vocab_size = create_dataloader(documents, batch_size)
    
    for idx, data in enumerate(train_loader):
        print('Shape of data:', data.shape)
        break
    
    model = CTM(num_topics, vocab_size, rho)
    
    # Hyper parameters for the model
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()

    # Model training
    train_model(model, num_epochs, train_loader, optimizer, criterion)
