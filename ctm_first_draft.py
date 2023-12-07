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
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.corpus     = None

        #Sufficient Statistics, used in m-phase of algorithm
        self.suf_num_docs   = 0
        self.suf_beta  = torch.zeros(num_topics,vocab_size)
        self.suf_mu    = torch.zeros(num_topics)
        self.suf_sigma = torch.zeros(num_topics, num_topics)


        #Model parameters

        #k-dimensional mean 
        self.mu = torch.zeros(self.num_topics)
        #covariance matrix of the model
        self.sigma = torch.diagflat(torch.tensor([1.0] * (self.num_topics)))
        #inverse of sigma - helpful for calculations later
        self.sigma_inv = torch.inverse(self.sigma)
        # beta: topic-word distribution
        self.beta = torch.rand(num_topics, vocab_size)

        #Debug check
        if(0 in self.beta):
            print("THERE IS A ZERO IN BETA 1")

        # self.phi = nn.Parameter(torch.randn(num_topics, num_topics))

        #variantional parameters

        #lamda and nusqrd are two indpendend univariate gaussians
        self.lamda = torch.zeros(self.num_topics)
        self.nusqrd = torch.ones(self.num_topics)

        # phi: topic-topic distribution
        #describes the variational distributions of topics 
        self.phi = 1/float(self.num_topics) * torch.ones(self.vocab_size, self.num_topics)
        
        #Zeta: variational parameter - appears when we upper bound the
        #log probability using a taylor expansion equation (7)
        self.zeta = self.max_zeta(self.lamda, self.nusqrd)

        self.corp_lamda = None #init when we get the corpus



    #Given a document this computes its log probability
    #This is equation (7) in the original CTM paper
    def log_bound(self, doc, nusqrd=None, lamda=None, phi=None):

        #we take lamda and nusqrd as optional parameters
        #this is so we can use this equation with an opmtimizer
        #like adam if we want to later
        if(lamda == None):
            lamda  = self.lamda
        
        if(nusqrd == None):
            nusqrd = self.nusqrd

        if(phi == None):
            phi = self.phi

        
        N = sum(doc) #number of words in doc

        #this is a debugging check 
        #If there is a zero we will get nan for the log bound later
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
      
        for n,c in enumerate(doc):

            for i in range(self.num_topics):

                #next two if statements are for debugging
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

                #another debuggin check
                if(torch.tensor(bound).isnan()):
                    print("Bound is Nan")
                    print("first: ", c*phi[n,i])
                    print("second: ", (lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i])))
                    print("beta: ", self.beta[i,n])
                    print("beta log: ", torch.log(self.beta[i,n]) )
                    print("phi log: ", torch.log(phi[n,i]))
                    exit()

        return bound

    #this computes the log bound for an entire corpus 
    def corpus_bound(self, corpus):
        return sum([self.log_bound(doc) for doc in corpus])

    #Training loop, runs until the log probability changes by at most TOL_EM
    #       OR we hit max iterations
    #default TOL values come from original CTM paper
    #MAX_ITERATIONS is just a value I set
    def train(self, corpus, TOL_E=10**-3, TOL_EM=10**-5, MAX_ITERATIONS=10000, stop=-1):
        # Two phases, e and then m, where we try to maximize bound on log prob
        # Repeat until difference is less than TOL_EM
        # e phase is coordinate ascent which runs till delta less than TOL_E
        self.corpus = corpus
        
        for i in range(MAX_ITERATIONS):
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

    #This implements the expectation (or E) step from the original paper
    #We maximize the log probability bound w.r.t. the variational parameters
    #It then updates the sufficient statistics which is used in the next phase
    def e_phase(self):

        print("Computing E Phase")
        self.clear_suf_stats()

        for i,doc in enumerate(self.corpus):
            if(i%4 == 0):
                print("Starting doc: ", i)

            #There are 4 parameters to maximize 
            self.zeta = self.max_zeta(self.lamda, self.nusqrd) 

            self.max_phi(self.beta, doc) 
            
            self.lamda = self.max_lamda(self.lamda, self.phi, doc)

            self.nusqrd = self.max_nusqrd(self.nusqrd, doc) 

            #update sufficient statistics for M step
            self.update_suf(doc)

    #MLE of the topics and multivariate gaussian
    #Called the M phase in original paper
    def m_phase(self):

        print("M PHASE")
    
        #debug statement if zero we will get a nan log bound
        if(0 in self.suf_beta):
            print("MPHASE ZERO")
            print(self.suf_beta)


        for i in range(self.num_topics):
            beta_norm  = torch.sum(self.suf_beta[i])
            self.beta[i]  = self.suf_beta[i] / beta_norm
        

        self.mu    = self.suf_mu/self.suf_num_docs
        self.sigma = self.suf_sigma + torch.matmul(self.mu, torch.t(self.mu))
        self.sigma_inv = torch.inverse(self.sigma)


    #Resets sufficient statistics 
    #Used at the start of the e phase
    def clear_suf_stats(self):
        self.suf_num_docs   = 0

        self.suf_beta       = torch.zeros(self.num_topics,self.vocab_size)
        self.suf_mu         = torch.zeros(self.num_topics)
        self.suf_sigma      = torch.zeros(self.num_topics, self.num_topics)


    #updates sufficient statistics, called during each iteration of e phase
    def update_suf(self, doc):

        self.suf_num_docs += 1
        self.suf_mu += self.lamda
        for n,c in enumerate(doc):
            for i in range(self.num_topics):                   
                self.suf_beta[i,n] += c * self.phi[n,i]
        
        self.suf_sigma += torch.diag(self.nusqrd) + torch.matmul(self.lamda, torch.t(self.lamda))

    #Maximize log bound wrt to zeta, this can be done with a closed form equation
    #Equation (14) of original paper
    def max_zeta(self, lamda, nusqrd):
        return torch.sum(torch.exp(torch.add(lamda,nusqrd,alpha=1/2)))
        

    #maximize log bound wrt to phi, again there is a closed equation
    #Comes from (15) of doc though honestly I'm not 100% on 
    #           the derivation of this, I referenced gensim for this
    def max_phi(self, beta, doc):

        for n,c in enumerate(doc):
            phi_norm = 0
            for i in range(self.num_topics):
                phi_norm = sum([torch.exp(self.lamda[i]) * beta[i,n]])
       
            for i in range(self.num_topics):
                self.phi[n,i] = torch.exp(self.lamda[i]) * self.beta[i,n]/phi_norm


    

    #This optmizes lamda, to maximize the log bound
    #sadly this can not be done analytically like that last 2
    #So I use adam to optimize. This is not how the original paper said
    #They used conjugate gradient descent, though I dont think that is an issue
    def max_lamda(self, lamda, phi, doc):
        
        def target(doc,lamda,phi):
            return self.log_bound(doc, lamda=lamda, phi=phi)


        opt = torch.optim.Adam([doc,lamda],maximize=True)#,differentiable=True)
        

        for i in range(10):
            opt.zero_grad()
            loss = target(doc, lamda, phi)
            # loss.backward()
            opt.step()

        return lamda
    
    #This optimizes nusqrd with respect to the log bound
    #This again like lamda is not how the paper did it, they used newtons method
    #I don't think thats an issue but its not hard to implement if we want to
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


#This creates a ctm and runs the training loop on data
#   Data - is a pandas dataframe of our documents
#num_doc is the number of documents in corpus and 
#num_topic is the number of topics to find
def runModel(data, num_doc, num_topics):

    print("Num topics", num_topics)
    documents = data.head(num_doc).content
    target_labels = data.head(num_doc).target
    target_names = data.head(num_doc).target_names

    num_epochs = 10

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
