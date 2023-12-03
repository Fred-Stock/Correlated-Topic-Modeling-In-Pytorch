import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ctm_dataloader import create_dataloader


#Freddy - Going to ignore hyper parameters for now 
#         Doing a strict implemenation of the original paper
class CTM():
    def __init__(self, num_topics, vocab_size):#, rho):
        super(CTM, self).__init__()

        # num_topics: number of topics in the model
        # vocab_size: size of the vocabulary
        # rho: hyperparameter for the model
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.suf_num_docs   = 0
        # self.rho = rho
        self.suf_beta  = torch.zeros(num_topics,vocab_size)
        # self.suf_phi = torch.zeros(num_topics,num_topics)
        self.suf_mu    = torch.zeros(num_topics)
        self.suf_sigma = torch.zeros(num_topics, num_topics)

        #These aren't quite the right initializations

        self.mu = torch.zeros(self.num_topics)
        self.sigma = torch.diagflat(torch.tensor([1.0] * (self.num_topics)))
        print(self.sigma)
        self.sigma_inv = torch.inverse(self.sigma)


        # beta: topic-word distribution
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))

        # phi: topic-topic distribution
        # self.phi = nn.Parameter(torch.randn(num_topics, num_topics))

        self.lamda = torch.zeros(self.num_topics)
        self.nusqrd = torch.ones(self.num_topics)
        self.phi = 1/float(self.num_topics) * torch.ones(self.vocab_size, self.num_topics)
        self.zeta = self.max_zeta(self.lamda, self.nusqrd)

        # mu: topic bias
        # self.mu = nn.Parameter(torch.randn(num_topics))

    def forward(self, x):
        # x is a batch of documents represented as a bag-of-words
        # x has shape (batch_size, vocab_size)
        batch_size = x.size(0)
        print(f"Batch size: {batch_size}")

        # log_prob: log probability of each document belonging to each topic
        log_prob = torch.zeros(batch_size, self.num_topics)
        print(f"Log probability {log_prob} and shape: {log_prob.shape}")
        for k in range(self.num_topics):
            print(f"Topic {k}")

            # compute the log probability of each document belonging to topic k
            log_prob[:, k] = torch.sum(x * self.beta[k], dim=1) + self.mu[k]

            for j in range(self.num_topics):
                if j != k:
                    # compute contribution of topic j to the log probability
                    # of each document belonging to topic k
                    log_prob[:, k] += (
                        self.phi[k, j] * torch.sum(x * self.beta[j], dim=1)
                    )

        return log_prob

    def log_bound(self, doc, nusqrd=None, lamda=None):

        #lamda and nusqrd as parameters enable use with an optimizer
        if(lamda == None):
            lamda  = self.lamda
        
        if(nusqrd == None):
            nusqrd = self.nusqrd

        print("doc", doc)
        N = sum([c for n,c in doc]) #this needs to be the number of words in doc

        bound  = 0.0

 
        bound += .5 * torch.log(torch.det(self.sigma_inv))
        bound -= .5 * torch.trace(torch.dot(torch.diag(nusqrd), self.sigma_inv))
        bound -= .5 * torch.t(lamda - self.mu).dot(self.sigma_inv).dot(lamda - self.mu)
        bound += .5 * (torch.sum(torch.log(nusqrd)) + self.num_topics)

        expect = torch.sum(torch.exp(lamda + (.5*nusqrd)))
        bound += (N * (-1/self.zeta * expect + 1 - torch.log(self.zeta)))

        #DEFINITELY FEEL LIKE THE STUFF BELOW MIGHT BE WRONG
        # for (n,c) in doc:
            # bound += torch.sum(c*self.phi[n] * (lamda + torch.log(torch.t(self.beta)[n])
                                                #  - torch.log(self.phi[n]))) 
        for n,c in doc:
            for i in range(self.num_topics):
                # self.suf_beta[i,n] += c * self.phi[n,i]
                bound += c*self.phi[n,i] * (lamda[i] + torch.log(self.beta[i,n] - torch.log(self.phi[n,i])))

        return bound

    def corpus_bound(self, corpus):
        return sum([self.log_bound(doc) for doc in corpus])

    #default TOL values come from original CTM paper
    #MAX_ITERATIONS is just a value I set
    def train(self, corpus, TOL_E=10**-6, TOL_EM=10**-5, MAX_ITERATIONS=10000):
        # Two phases, e and then m, where we try to maximize bound on log prob
        # Repeat until difference is less than TOL_EM
        # e phase is coordinate ascent which runs till delta less than TOL_E
        
        for i in range(MAX_ITERATIONS):
            old_bound = self.corpus_bound(corpus)
            self.e_phase()
            self.m_phase()

            delta = self.corpus_bound(corpus) - old_bound

            if delta < TOL_EM:
                break

    def e_phase(self):
        
        #There are 4 parameters to maximize 
        self.zeta = self.max_zeta(self.lamda, self.nusqrd) #analytical

        self.phi = self.max_phi(self.lamda, self.beta) #analytical 

        self.lamda = self.max_lamda(self.lamda) #conjugate gradient descent

        self.nusqrd = self.max_nusqrd(self.nusqrd) #newtons method

        #have to update sufficient statistics so we can run the M step

    def m_phase(self):#MLE

        beta_norm  = torch.sum(self.suf_beta)
        self.beta  = self.suf_beta / beta_norm

        self.mu    = self.suf_mu/self.suf_num_docs
        self.sigma = self.suf_sigma + torch.multiply(self.mu, torch.t(self.mu))
        self.sigma_inv = torch.inv(self.sigma)

        return "NOT DEFINED"
    
    def update_suf(self, doc):

        self.suf_num_docs += 1
        #update sufficient statistics
        self.suf_mu += self.lamda
        for n,c in doc:
            for i in range(self.num_topics):
                self.suf_beta[i,n] += c * self.phi[n,i]
        
        self.suf_sigma += torch.diag(self.nusqrd) + torch.dot(self.lamda, torch.t(self.lamda))

    def max_zeta(self, lamda, nusqrd):
        return torch.sum(torch.exp(torch.add(lamda,nusqrd,alpha=1/2)))
        

    def max_phi(self, phi, beta):

        expP = torch.exp(phi)
        prod = torch.sum(expP * beta, dim=1)
        
        return prod/torch.sum(prod)
    
    #The two below aren't quite how the paper handled it
    #But this seemed like the path of least resistance
    #Hopefully it works
    def max_lamda(self, lamda):
        
        def target(lamda):
            return self.log_bound(lamda)

        # def deriv(lamda):
            

        opt = torch.optim.Adam(lamda,maximize=True)#,differentiable=True)
        

        for i in range(10):
            opt.zero_grad()
            loss = target(lamda)
            loss.backward()
            opt.step()

        return lamda
    
    def max_nusqrd(self, nusqrd):
        
        def target(nusqrd):
            return self.log_bound(nusqrd)

        # def deriv(lamda):
            

        opt = torch.optim.Adam(nusqrd,maximize=True)#,differentiable=True)
        

        for i in range(10):
            opt.zero_grad()
            loss = target(nusqrd)
            loss.backward()
            optimizer.step()

        return nusqrd




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
    # print(train_loader[0])
    print(documents)

    for idx, data in enumerate(train_loader):
        print("Shape of data:", data.shape)
        break

    # print(train_loader[0])
    

    model = CTM(num_topics, vocab_size)
    model.train(train_loader)

    # Hyper parameters for the model
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    # criterion = nn.CrossEntropyLoss()

    # Model training
    # train_model(model, num_epochs, train_loader, optimizer, criterion)
