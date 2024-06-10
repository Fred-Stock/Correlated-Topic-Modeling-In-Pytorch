import pandas as pd
import numpy as np
import torch
from ctm_dataloader import create_dataloader
import seaborn as sns
import matplotlib.pyplot as plt
import copy


class SufficientStats():
    """
    Stores statistics about variational parameters during E-step in order
    to update CtmModel's parameters in M-step.

    `self.mu_stats` contains sum(lamda_d)

    `self.sigma_stats` contains sum(I_nu^2 + lamda_d * lamda^T)

    `self.beta_stats[i]` contains sum(phi[d, i] * n_d) where nd is the vector
    of word counts for document d.

    `self.numtopics` contains the number of documents the statistics are build on

    """

    def __init__(self, num_topics, vocab_size):
        self.num_docs = 0
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.beta_stats = torch.zeros([num_topics, vocab_size])
        self.mu_stats = torch.zeros(num_topics)
        self.sigma_stats = torch.zeros([num_topics, num_topics])

    def update(self, lamda, nu2, phi, doc):
        """
        Given optimized variational parameters, update statistics

        """

        # update mu_stats
        self.mu_stats += lamda

        # update \beta_stats[i], 0 < i < self.numtopics
        for n, c in enumerate(doc):
            for i in range(self.num_topics):
                self.beta_stats[i, n] += c * phi[n, i]
                
        # update \sigma_stats
        self.sigma_stats += torch.diag(nu2) + torch.matmul(lamda, torch.t(lamda))

        self.num_docs += 1

class CTM():
    
    def __init__(self, num_topics, vocab_size,
                 tol_em = 1e-6, tol_es = 1e-5, max_iter = 10):
        super(CTM, self).__init__()

        # num_topics: number of topics in the model
        # vocab_size: size of the vocabulary
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.tol_em = tol_em # relative change we need to achieve in E-step
        self.tol_es = tol_es # relative change we need to achieve in Expectation-Maximization
        self.max_iter = max_iter

        # Model parameters

        #k-dimensional mean 
        self.mu = torch.zeros(self.num_topics)
        #covariance matrix of the model
        self.sigma = torch.eye(self.num_topics)
        #inverse of sigma - helpful for calculations later
        self.sigma_inv = torch.inverse(self.sigma)
        # beta: topic-word distribution
        self.beta = torch.rand(num_topics, vocab_size)
        # theta: document-topic distribution
        self.theta = torch.rand(num_topics)

        #Debug check
        if(0 in self.beta):
            print("THERE IS A ZERO IN BETA 1")

        # Variantional parameters

        # lamda and nusqrd are two indpendend univariate gaussians
        self.lamda = torch.zeros(self.num_topics)
        self.nusqrd = torch.ones(self.num_topics)

        # phi: topic-topic distribution
        # describes the variational distributions of topics 
        self.phi = 1/float(self.num_topics) * torch.ones(self.vocab_size, self.num_topics)
        
        # Zeta: variational parameter - appears when we upper bound the
        # log probability using a taylor expansion equation (7)
        self.zeta = self.max_zeta()
        
    def train(self, corpus):
        # Two phases, e and then m, where we try to maximize bound on log prob
        # Repeat until difference is less than TOL_EM
        # e phase is coordinate ascent which runs till delta less than TOL_E
        
        for i in range(self.max_iter):
            print(f'----- Training epoch {i+1} of {self.max_iter} -----')
            
            old_bound = self.corpus_bound(corpus)
            print(f'Initial bound at the start of train_loop: {old_bound}')
            
            statistics = self.e_phase(corpus)
            self.m_phase(statistics)

            delta = (self.corpus_bound(corpus) - old_bound)/old_bound

            print("delta",delta)
            print("non-normalized delta", (delta * old_bound))

            # max just to make sure delta is positive, could use abs
            if max(-delta, delta) < self.tol_em:
                break
            
    def log_bound(self, doc, lamda=None, nusqrd=None, phi=None):
        """ Compute the bound on the log probability of the corpus """
        if lamda is None:
            lamda = self.lamda
        if nusqrd is None:
            nusqrd = self.nusqrd
        if phi is None:
            phi = self.phi
            
        N = sum(doc)
        bound  = 0.0
        
        # Part 1 of equation 4
        bound += 0.5 * torch.log(torch.linalg.det(self.sigma_inv))
        bound += -0.5 * self.num_topics * np.log(2*np.pi)
        bound += -0.5 * torch.trace(torch.diag(self.nusqrd) @ self.sigma_inv) \
            + torch.t(self.lamda - self.mu) @ self.sigma_inv @ (self.lamda - self.mu)
           
        # Part 2 partial of equation 4 
        """Eq [log p(zn | η)]"""
        expect = torch.sum(torch.exp(self.lamda + 0.5*self.nusqrd)).item()
        bound += (N * (-1/self.zeta * expect + 1 - torch.log(self.zeta))).item()

        # Part 4 partial of equation 4
        """Entropy term"""
        sum_1 = 0
        for topic in range(self.num_topics):
            sum_1 += torch.log(self.nusqrd[topic]) + np.log(2 * np.pi) + 1
            
        bound += sum_1.item()
        
        # Part 2 partial, 3 and 4 partial of equation 4
        for n, c in enumerate(doc):
            for i in range(self.num_topics):
                bound += (c*phi[n,i] * (lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i]))).item()
                
                # debugging check 1
                if(torch.tensor(bound).isnan()):
                    print("Bound is Nan")
                    print("first: ", c*phi[n,i])
                    print("second: ", (lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i])))
                    # print("beta: ", self.beta[i])
                    print("beta: ", self.beta[i,n])
                    print("beta log: ", torch.log(self.beta[i,n]))
                    print("phi log: ", torch.log(phi[n,i]))
                    exit()
                    
                # debugging check 2
                if((c*phi[n,i]).isnan()):
                    print("first isNan")
                    print("phi", phi)
                    print("element",phi[n,i])
                
                # debugging check 3
                if ((lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i]))).isnan():
                    print("second isNan")
                    print(lamda[i])
                    print(self.beta[i,n])
                    print(phi[n,i])
                    exit()
        
        
        # bound = 0.0
        # bound += self.e_phase_part1()
        # if bound.isnan():
        #     print("part 1 is nan")
        # bound += self.e_phase_part2(doc)
        # if bound.isnan():
        #     print("part 2 is nan")
        # bound += self.e_phase_part3(doc)
        # if bound.isnan():
        #     print("part 3 is nan")
        # bound += self.e_phase_part4()
        # if bound.isnan():
        #     print("part 4 is nan")
        # bound += .5 * torch.log(torch.det(self.sigma_inv)).item()
        # bound -= .5 * torch.trace(torch.matmul(
        #     torch.diag(nusqrd), self.sigma_inv)).item()

        # bound -= .5 * torch.t(self.lamda - self.mu
        #                       ).matmul(self.sigma_inv
        #                                  ).matmul(self.lamda - self.mu).item()     

        # bound += .5 * (torch.sum(torch.log(nusqrd)) + self.num_topics).item()

        # expect = torch.sum(torch.exp(lamda + (.5*nusqrd))).item()
        # bound += (N * (-1/self.zeta * expect + 1 - torch.log(self.zeta))).item()

        # for n,c in enumerate(doc):

        #     for i in range(self.num_topics):

        #         #next two if statements are for debugging
        #         if((c*phi[n,i]).isnan()):
        #             print("first isNan")
        #             print("phi", phi)
        #             print("element",phi[n,i])

        #         if ((lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i]))).isnan():
        #             print("second isNan")
        #             print(lamda[i])
        #             print(self.beta[i,n])
        #             print(phi[n,i])
        #             exit()

        #         bound += (c*phi[n,i] * (lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i]))).item()

        #         #another debugging check
        #         if(torch.tensor(bound).isnan()):
        #             print("Bound is Nan")
        #             print("first: ", c*phi[n,i])
        #             print("second: ", (lamda[i] + torch.log(self.beta[i,n]) - torch.log(phi[n,i])))
        #             print("beta: ", self.beta[i,n])
        #             print("beta log: ", torch.log(self.beta[i,n]) )
        #             print("phi log: ", torch.log(phi[n,i]))
        #             exit()
        
        return bound
    
    def corpus_bound(self, corpus):
        """ Compute the bound on the log probability of the corpus """
        return sum([self.log_bound(doc) for doc in corpus])
            
    def e_phase(self, corpus):
        """Coordinate ascent optimization of the variational parameters"""
        print('Starting E-phase')
        statistics = SufficientStats(self.num_topics, self.vocab_size)
        
        for i, doc in enumerate(corpus):
            if i % 8 == 0:
                print(f'Processing document {i+1} to {i+8} of {len(corpus)} documents')
            
            model = copy.deepcopy(self)
            model.variational_inference(doc)
            
            statistics.update(self.lamda, self.nusqrd, self.phi, doc)
        print(f'Beta at the end of e_phase: {self.beta.shape}')
        #Debug check
        if(0 in self.beta):
            print("THERE IS A ZERO IN BETA 1")
            
        return statistics
    
    def variational_inference(self, doc):
        bound = self.log_bound(doc)
        new_bound = bound
        
        for _ in range(self.max_iter):
            self.max_zeta()
            self.max_phi(doc)
            self.max_lamda(doc, self.lamda, self.phi)
            self.max_nusqrd(doc, self.nusqrd)
            
            # bound, new_bound = new_bound, self.log_bound(doc)
            # relative_change = abs((new_bound - bound) / bound)
            
            # if (relative_change < self.tol_es):
            #     break
        
    def e_phase_part1(self):
        """Eq [log p(η | µ, Σ)]"""
        part_1 = 0.5 * torch.log(torch.linalg.det(self.sigma_inv))
        part_2 = -0.5 * self.num_topics * np.log(2*np.pi)
        part_3 = -0.5 * torch.trace(torch.diag(self.nusqrd) @ self.sigma_inv) \
            + torch.t(self.lamda - self.mu) @ self.sigma_inv @ (self.lamda - self.mu)
        return part_1 + part_2 + part_3
    
    def e_phase_part2(self, doc):
        """Eq [log p(zn | η)]"""
        part_1 = 0.0
        for n,c in enumerate(doc):
            for i in range(self.num_topics):
                part_1 += c*self.phi[n,i] * self.lamda[i]
        part_2 = -(1/self.zeta) * torch.sum(torch.exp(self.lamda + 0.5*self.nusqrd)) + 1 - torch.log(self.zeta)
        return part_1 + part_2
    
    def e_phase_part3(self, doc):
        """Eq [log p(wn | zn, β)]"""
        sum = 0
        for n,c in enumerate(doc):
            for i in range(self.num_topics):
                sum += c * torch.sum(self.phi[n,i] * torch.log(self.beta[i,n]))
        return sum
    
    def e_phase_part4(self):
        sum_1 = 0
        for topic in range(self.num_topics):
            sum_1 += torch.log(self.nusqrd[topic]) + np.log(2 * np.pi) + 1
        sum_2 = 0
        for word in range(self.vocab_size):
            for topic in range(self.num_topics):
                sum_2 += self.phi[word, topic] * torch.log(self.phi[word, topic])
        return 0.5 * sum_1 - sum_2
    
    def max_zeta(self):
        """Maximize zeta"""
        return torch.sum(torch.exp(torch.add(self.lamda, self.nusqrd, alpha = 0.5)))
    
    def max_phi(self, corpus):
        phi_norm = 0.0
        for n, _ in enumerate(corpus):
            for i in range(self.num_topics):
                phi_norm = sum([torch.exp(self.lamda[i]) * self.beta[i,n]])
            for i in range(self.num_topics):
                self.phi[n, i] = torch.exp(self.lamda[i]) * self.beta[i, n] / phi_norm
                
    def max_lamda(self, doc, lamda, phi):
        
        def target(doc, lamda, phi):
            return self.log_bound(doc, lamda = lamda, phi = phi)
        
        optimizer = torch.optim.Adam([doc, lamda], maximize = True)
        
        for _ in range(10):
            optimizer.zero_grad()
            loss = target(doc, lamda, phi)
            # loss.backward()
            optimizer.step()
            
    
    def max_nusqrd(self, doc, nusqrd):
        
        def target(doc, nusqrd):
            return self.log_bound(doc, nusqrd = nusqrd)
        
        optimizer = torch.optim.Adam([doc, nusqrd], maximize = True)
        
        for _ in range(10):
            optimizer.zero_grad()
            loss = target(doc, nusqrd)
            # loss.backward()
            optimizer.step()
            
    def m_phase(self, sstats):
        """Maximize the parameters of the model"""
        print('Starting M-phase')
        if(0 in sstats.beta_stats):
            print("MPHASE ZERO")
            # print(sstats.beta_stats)
        
        for i in range(self.num_topics):
            beta_norm = torch.sum(sstats.beta_stats[i])
            self.beta[i] = sstats.beta_stats[i] / beta_norm
        
        self.mu = sstats.mu_stats / sstats.num_docs
        self.sigma = sstats.sigma_stats + torch.matmul(self.mu, torch.t(self.mu))
        self.sigma_inv = torch.inverse(self.sigma)
        print(f'Beta at the end of m_phase: {self.beta.shape}')
        #Debug check
        if(0 in self.beta):
            print("THERE IS A ZERO IN BETA 1")

#num_topic is the number of topics to find
def runModel(data, num_topics):

    batch_size = 32
    lr = 1e-5

    # Create dataloader
    train_loader, vocab_size = create_dataloader(data, batch_size)

    torch.set_printoptions(threshold=20000) #lets us print larger tensors to terminal for debugging

    model = CTM(num_topics, vocab_size)
    
    
    for i, batch in enumerate(train_loader):
        print(f'--------- Training batch {i+1} ---------')
        model.train(batch)
        
    print(f'Final probabilies: {model.beta}')
    
    # save the model.beta
    torch.save(model.beta, "beta.pt")
    
    # corrCoef = torch.corrcoef(model.beta)
    # sns.heatmap(corrCoef, annot=True)
    # plt.show(block=False)

if __name__ == "__main__":
    
    news_data = pd.read_csv('newsgroups_data.csv')
    news_data = news_data.drop(columns=["Unnamed: 0"])
    
    num_doc = 64
    
    documents = news_data.head(num_doc).content
    target_labels = news_data.head(num_doc).target
    
    runModel(documents, target_labels.nunique())
    
