import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from ctm_dataloader import create_dataloader
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.corpus     = None
        self.suf_num_docs   = 0
        # self.rho = rho
        self.suf_beta  = torch.zeros(num_topics,vocab_size)
        # self.suf_phi = torch.zeros(num_topics,num_topics)
        self.suf_mu    = torch.zeros(num_topics)
        self.suf_sigma = torch.zeros(num_topics, num_topics)


        self.mu = torch.zeros(self.num_topics)
        self.sigma = torch.diagflat(torch.tensor([1.0] * (self.num_topics)))
        print(self.sigma)
        self.sigma_inv = torch.inverse(self.sigma)


        # beta: topic-word distribution
        self.beta = torch.rand(num_topics, vocab_size)
        # print(self.beta)
        # exit()
        
        if(0 in self.beta):
            print("THERE IS A ZERO IN BETA 1")

        # phi: topic-topic distribution
        # self.phi = nn.Parameter(torch.randn(num_topics, num_topics))

        self.lamda = torch.zeros(self.num_topics)
        self.nusqrd = torch.ones(self.num_topics)
        self.phi = 1/float(self.num_topics) * torch.ones(self.vocab_size, self.num_topics)
        self.zeta = self.max_zeta(self.lamda, self.nusqrd)

        self.corp_lamda = None #init when we get the corpus

        # mu: topic bias
        # self.mu = nn.Parameter(torch.randn(num_topics))

    # def forward(self, x):
    #     # x is a batch of documents represented as a bag-of-words
    #     # x has shape (batch_size, vocab_size)
    #     batch_size = x.size(0)
    #     print(f"Batch size: {batch_size}")

    #     # log_prob: log probability of each document belonging to each topic
    #     log_prob = torch.zeros(batch_size, self.num_topics)
    #     print(f"Log probability {log_prob} and shape: {log_prob.shape}")
    #     for k in range(self.num_topics):
    #         print(f"Topic {k}")

    #         # compute the log probability of each document belonging to topic k
    #         log_prob[:, k] = torch.sum(x * self.beta[k], dim=1) + self.mu[k]

    #         for j in range(self.num_topics):
    #             if j != k:
    #                 # compute contribution of topic j to the log probability
    #                 # of each document belonging to topic k
    #                 log_prob[:, k] += (
    #                     self.phi[k, j] * torch.sum(x * self.beta[j], dim=1)
    #                 )

    #     return log_prob

    def log_bound(self, doc, nusqrd=None, lamda=None, phi=None):

        # print("in bound")
        #lamda and nusqrd as parameters enable use with an optimizer
        if(lamda == None):
            lamda  = self.lamda
        
        if(nusqrd == None):
            nusqrd = self.nusqrd

        if(phi == None):
            phi = self.phi

        # print("doc", doc)
        N = sum(doc) #this needs to be the number of words in doc

        if(0 in self.beta):
            print("BOUND THERE IS A ZERO IN BETA")
            exit()

        bound  = 0.0

 
        bound += .5 * torch.log(torch.det(self.sigma_inv)).item()
        
        if(torch.tensor(bound).isnan()):
            print("Bound is Nan1")
            exit()

        bound -= .5 * torch.trace(
            torch.matmul(
                torch.diag(nusqrd), self.sigma_inv)).item()
        
        if(torch.tensor(bound).isnan()):
            print("Bound is Nan2")
            exit()
        
        bound -= .5 * torch.t(self.lamda - self.mu
                              ).matmul(self.sigma_inv
                                         ).matmul(self.lamda - self.mu).item()

        if(torch.tensor(bound).isnan()):
            print("Bound is Nan3")  
            exit()      

        bound += .5 * (torch.sum(torch.log(nusqrd)) + self.num_topics).item()

        if(torch.tensor(bound).isnan()):
            print("Bound is Nan4")
            exit()

        # print("first half Computed bound")
        expect = torch.sum(torch.exp(lamda + (.5*nusqrd))).item()

        if(torch.tensor(bound).isnan()):
            print("Bound is Nan5")
            exit()

        bound += (N * (-1/self.zeta * expect + 1 - torch.log(self.zeta))).item()
        # print("Bound is done")

        if(torch.tensor(bound).isnan()):
            print("Bound is Nan6")
            exit()
        #DEFINITELY FEEL LIKE THE STUFF BELOW MIGHT BE WRONG
        # for (n,c) in doc:
            # bound += torch.sum(c*self.phi[n] * (lamda + torch.log(torch.t(self.beta)[n])
                                                #  - torch.log(self.phi[n]))) 
        
        # print(lamda)
        for n,c in enumerate(doc):
            # if(n%500 == 0):
                # print("doc enumeration")

            for i in range(self.num_topics):

                if((c*phi[n,i]).isnan()):
                    print("first isNan")
                    print("phi", phi)
                    print("element",phi[n,i])

                if ((lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i]))).isnan():
                    print("second isNan")
                    print(lamda[i])
                    print(self.beta[i,n])
                    print(phi[n,i])
                    exit()

                bound += (c*phi[n,i] * (lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i]))).item()

                if(torch.tensor(bound).isnan()):
                    print("Bound is Nan")
                    print("first: ", c*phi[n,i])
                    print("second: ", (lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i])))
                    print("beta: ", self.beta[i,n])
                    print("beta log: ", torch.log(self.beta[i,n]) )
                    print("phi log: ", torch.log(phi[n,i]))
                    exit()

        return bound

    def corpus_bound(self, corpus):
        return sum([self.log_bound(doc) for doc in corpus])

    #default TOL values come from original CTM paper
    #MAX_ITERATIONS is just a value I set
    def train(self, corpus, TOL_E=10**-3, TOL_EM=10**-5, MAX_ITERATIONS=10000, stop=-1):
        # Two phases, e and then m, where we try to maximize bound on log prob
        # Repeat until difference is less than TOL_EM
        # e phase is coordinate ascent which runs till delta less than TOL_E
        self.corpus = corpus
        # if(stop==1):
        #     print(self.phi)
        #     exit()
        # self.corp_lamda = torch.zeros(len(corpus)) might need this to get topic graph

        for i in range(MAX_ITERATIONS):
            # if(i%20 == 0):
            print("training loop ",i)
            old_bound = self.corpus_bound(corpus)
            print(old_bound)
            self.e_phase()
            self.m_phase()


            delta = (self.corpus_bound(self.corpus) - old_bound)/old_bound

            print("delta",delta)
            print("non-normalized delta", (delta*old_bound))

            #max just to make sure delta is positive, could use abs
            if max(-delta,delta) < TOL_EM:
                # print(self.beta)
                break

    def e_phase(self):
        #this phase maximizes variational parameters and then computes sufficient statistics 
        #   for MLE in the next phase

        print("Computing E Phase")
        self.clear_suf_stats()

        for i,doc in enumerate(self.corpus):
            if(i%4 == 0):
                print("Starting doc: ", i)
            #There are 4 parameters to maximize 
            self.zeta = self.max_zeta(self.lamda, self.nusqrd) #analytical

            self.max_phi(self.beta, doc) #analytical 
            # print(self.phi)
            self.lamda = self.max_lamda(self.lamda, self.phi, doc) #conjugate gradient descent

            self.nusqrd = self.max_nusqrd(self.nusqrd, doc) #newtons method

            #have to update sufficient statistics so we can run the M step
            self.update_suf(doc)

    def m_phase(self):#MLE

        print("M PHASE")
    
        if(0 in self.suf_beta):
            print("MPHASE ZERO")
            print(self.suf_beta)


        for i in range(self.num_topics):
            beta_norm  = torch.sum(self.suf_beta[i])
            self.beta[i]  = self.suf_beta[i] / beta_norm
        

        self.mu    = self.suf_mu/self.suf_num_docs
        self.sigma = self.suf_sigma + torch.matmul(self.mu, torch.t(self.mu))
        self.sigma_inv = torch.inverse(self.sigma)


    def clear_suf_stats(self):
        self.suf_num_docs   = 0
        # self.rho = rho
        self.suf_beta       = torch.zeros(self.num_topics,self.vocab_size)
        # self.suf_phi = torch.zeros(num_topics,num_topics)
        self.suf_mu         = torch.zeros(self.num_topics)
        self.suf_sigma      = torch.zeros(self.num_topics, self.num_topics)


    def update_suf(self, doc):

        self.suf_num_docs += 1
        #update sufficient statistics
        self.suf_mu += self.lamda
        for n,c in enumerate(doc):
            for i in range(self.num_topics):                   
                self.suf_beta[i,n] += c * self.phi[n,i]
        
        self.suf_sigma += torch.diag(self.nusqrd) + torch.matmul(self.lamda, torch.t(self.lamda))

    def max_zeta(self, lamda, nusqrd):
        return torch.sum(torch.exp(torch.add(lamda,nusqrd,alpha=1/2)))
        

    def max_phi(self, beta, doc):

        for n,c in enumerate(doc):
            phi_norm = 0
            for i in range(self.num_topics):
                phi_norm = sum([torch.exp(self.lamda[i]) * beta[i,n]])
                if(phi_norm == 0):
                    print("zero! ", i)
                    print(self.lamda[i])
                    print(self.lamda)
                    print(beta[i,n])
                    print(beta)
                    exit()


            
            for i in range(self.num_topics):
                self.phi[n,i] = torch.exp(self.lamda[i]) * self.beta[i,n]/phi_norm
                if (torch.exp(self.lamda[i]) * self.beta[i,n]/phi_norm).isnan():
                    print("NAN!")
                    print(self.lamda[i])
                    print(self.beta[i,n])


    
    #The two below aren't quite how the paper handled it
    #But this seemed like the path of least resistance
    #Hopefully it works
    def max_lamda(self, lamda, phi, doc):
        
        def target(doc,lamda,phi):
            return self.log_bound(doc, lamda=lamda, phi=phi)

        # def deriv(lamda):

        # print(phi.shape)
            

        opt = torch.optim.Adam([doc,lamda],maximize=True)#,differentiable=True)
        

        for i in range(10):
            opt.zero_grad()
            loss = target(doc, lamda, phi)
            # loss.backward()
            opt.step()

        return lamda
    
    def max_nusqrd(self, nusqrd, doc):
        
        def target(nusqrd, doc):
            return self.log_bound(doc, nusqrd=nusqrd)

        # def deriv(lamda):
            

        opt = torch.optim.Adam([doc, nusqrd],maximize=True)#,differentiable=True)
        

        for i in range(10):
            opt.zero_grad()
            loss = target(nusqrd, doc)
            # loss.backward()
            opt.step()

        return nusqrd


def runModel(data, num_doc, num_topics):

    print("Num topics", num_topics)
    documents = data.head(num_doc).content
    target_labels = data.head(num_doc).target
    target_names = data.head(num_doc).target_names

    # Hyperparameters
    num_epochs = 10
    # print("topics:", num_topics)
    # exit()
    batch_size = num_doc
    rho = 0.1
    lr = 0.001

    # Create dataloader
    train_loader, vocab_size = create_dataloader(documents, batch_size)


    torch.set_printoptions(threshold=20000) #lets us print larger tensors to terminal for debugging


    model = CTM(num_topics, vocab_size)
    
    
    for i, batch in enumerate(train_loader):
        print("Training batch: ", i)
        model.train(batch,stop=i)

    corrCoef = torch.corrcoef(model.beta)
    sns.heatmap(corrCoef, annot=True)
    plt.show(block=False)



if __name__ == "__main__":
    # Load the dataset

    news_data = pd.read_csv("newsgroups_data.csv")
    print("Shape of dataset:", news_data.shape)

    news_data = news_data.drop(columns=["Unnamed: 0"])


    # runModel(news_data, 10, 10)
    # runModel(news_data, 20, 10)
    # runModel(news_data, 20, 20)
    
    runModel(news_data, 64, 20)

    plt.show()



    # Hyper parameters for the model
    # optimizer = optim.Adam(model.parameters(), lr = lr)
    # criterion = nn.CrossEntropyLoss()

    # Model training
    # train_model(model, num_epochs, train_loader, optimizer, criterion)
