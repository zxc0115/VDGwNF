import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
import numpy as np
import random
import sys
#CurrentPath = os.path.abspath(".")
#print(CurrentPath)
#sys.path.insert(0, CurrentPath)
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import gVar, gData
from modules import Encoder, ContextEncoder, Variation, Decoder, mean_zero_Variation           
import flows as flow
import nn as nn_
one = gData(torch.FloatTensor([1]))
minus_one = one * -1    
def log_Normal_diag(x, mean, log_var, average=True, dim=1):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var ), -1) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )
class DFVAE(nn.Module):
    def __init__(self, config, vocab_size, PAD_token=0):
        super(DFVAE, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen=config['maxlen']
        self.clip = config['clip']
        self.lambda_gp = config['lambda_gp']
        self.temp=config['temp']
        
        self.embedder= nn.Embedding(vocab_size, config['emb_size'], padding_idx=PAD_token)
        self.utt_encoder = Encoder(self.embedder, config['emb_size'], config['n_hidden'], 
                                   True, config['n_layers'], config['noise_radius']) 
        self.context_encoder = ContextEncoder(self.utt_encoder, config['n_hidden']*2+2, config['n_hidden'], 1, config['noise_radius']) 
        self.prior_net = Variation(config['n_hidden'], config['z_size']) # p(e|c)
        self.post_net = Variation(config['n_hidden']*3, config['z_size']) # q(e|c,x)
        
        #self.prior_highway = nn.Linear(config['n_hidden'], config['n_hidden'])
        #self.post_highway = nn.Linear(config['n_hidden'] * 3, config['n_hidden'])
        self.postflow1 = flow.myIAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        self.postflow2 = flow.myIAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        self.postflow3 = flow.myIAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        self.priorflow1 = flow.IAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        self.priorflow2 = flow.IAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        self.priorflow3 = flow.IAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        
        self.post_generator = nn_.SequentialFlow(self.postflow1,self.postflow2,self.postflow3)
        self.prior_generator = nn_.SequentialFlow(self.priorflow1,self.priorflow2,self.priorflow3)
                                                                                             
        self.decoder = Decoder(self.embedder, config['emb_size'], config['n_hidden']+config['z_size'], 
                               vocab_size, n_layers=1) 
           
        self.optimizer_AE = optim.SGD(list(self.context_encoder.parameters())
                                      +list(self.post_net.parameters())
                                      +list(self.post_generator.parameters())
                                      +list(self.decoder.parameters())
                                      +list(self.prior_net.parameters())
                                      +list(self.prior_generator.parameters())
                                      #+list(self.prior_highway.parameters())
                                      #+list(self.post_highway.parameters())
                                      ,lr=config['lr_ae'])
        self.optimizer_G = optim.RMSprop(list(self.post_net.parameters())
                                      +list(self.post_generator.parameters())
                                      +list(self.prior_net.parameters())
                                      +list(self.prior_generator.parameters())
                                      #+list(self.prior_highway.parameters())
                                      #+list(self.post_highway.parameters())
                                      , lr=config['lr_gan_g'])
        
        #self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=config['lr_gan_d'])
        
        self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer_AE, step_size = 10, gamma=0.6)
        
        self.criterion_ce = nn.CrossEntropyLoss()
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)
    def sample_post(self, x, c):
        xc = torch.cat((x, c),1)
        e, mu, log_s = self.post_net(xc)
        #h_post = self.post_highway(xc)
        z, det_f,_,_ = self.post_generator((e,torch.eye(e.shape[1]), c, mu))
        #h_prior = self.prior_highway(c)
        tilde_z, det_g, _ = self.prior_generator((z, det_f, c))
        return tilde_z, z, mu, log_s, det_f, det_g 
    def sample_code_post(self, x, c):
        xc = torch.cat((x, c),1)
        e, mu, log_s = self.post_net(xc)
        #h_post = self.post_highway(xc)
        z, det_f,_,_ = self.post_generator((e, torch.eye(e.shape[1]), c, mu))
        #h_prior = self.prior_highway(c)
        tilde_z, det_g, _ = self.prior_generator((z, det_f, c))
        return tilde_z, mu, log_s, det_f, det_g
    def sample_post2(self, x, c):
        xc = torch.cat((x, c),1)
        e, mu, log_s = self.post_net(xc)
        #h_post = self.post_highway(xc)
        z, det_f,_,_ = self.post_generator((e, torch.eye(e.shape[1]), c, mu))
        return e, mu, log_s, z , det_f
   
    def sample_code_prior(self, c):
        e, mu, log_s = self.prior_net(c)
        #z = self.prior_generator(e)
        #h_prior = self.prior_highway(c)
        #tilde_z, det_g, _ = self.prior_generator((e, 0, h_prior))
        return e, mu, log_s#, det_g  
    def sample_prior(self,c):
        e, mu, log_s = self.prior_net(c)
        #h_prior = self.prior_highway(c)
        z, det_prior, _ = self.prior_generator((e,0, c))
        return z, det_prior
    def train_AE(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.train()
        self.decoder.train()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)      
        z, _, _, _, _ = self.sample_code_post(x, c)
        z_post, mu_post, log_s_post, det_f, det_g = self.sample_code_post(x, c)
        #prior_z, mu_prior, log_s_prior = self.sample_code_prior(c)
        #KL_loss = torch.sum(log_s_prior - log_s_post + (torch.exp(log_s_post) + (mu_post - mu_prior)**2)/torch.exp(log_s_prior),1) / 2 - 100
        #kloss = KL_loss - det_f #+ det_g
        #KL_loss = log_Normal_diag(z_post, mu_post, log_s_post) - log_Normal_diag(prior_z, mu_prior, log_s_prior)
        output = self.decoder(torch.cat((z_post, c),1), None, response[:,:-1], (res_lens-1))  
        flattened_output = output.view(-1, self.vocab_size) 
        
        dec_target = response[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) # 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)# [(batch_sz*seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        #print(KL_loss.mean())
        #print(det_f.mean())
        self.optimizer_AE.zero_grad()
        AE_term = self.criterion_ce(masked_output/self.temp, masked_target)
        loss = AE_term #+ KL_loss.mean()
        loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(list(self.context_encoder.parameters())+list(self.decoder.parameters()), self.clip)
        torch.nn.utils.clip_grad_norm_(list(self.context_encoder.parameters())+list(self.decoder.parameters())+list(self.post_generator.parameters())+list(self.prior_generator.parameters())+list(self.post_net.parameters()), self.clip)
        self.optimizer_AE.step()

        return [('train_loss_AE', AE_term.item())]#,('KL_loss', KL_loss.mean().item())]#,('det_f', det_f.mean().item()),('det_g', det_g.mean().item())]        
    
    def train_G(self, context, context_lens, utt_lens, floors, response, res_lens): 
        self.context_encoder.eval()
        self.optimizer_G.zero_grad()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        # -----------------posterior samples ---------------------------
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        z_0, mu_post, log_s_post, z_post, weight = self.sample_post2(x.detach(), c.detach())
        # ----------------- prior samples ---------------------------
        prior_z, mu_prior, log_s_prior = self.sample_code_prior(c.detach())
        KL_loss = torch.sum(log_s_prior - log_s_post + torch.exp(log_s_post)/torch.exp(log_s_prior) * torch.sum(weight**2,dim=2) + (mu_post)**2/torch.exp(log_s_prior),1) / 2 - 100 
        #KL_loss = abs(log_Normal_diag(z_0, mu_post, log_s_post) - log_Normal_diag(z_post, mu_prior, log_s_prior))
        #KL_loss2 = torch.sum((prior_z - mu_post.detach())**2 / (2 * torch.exp(log_s_post.detach())),1)
        #print(mu_post.shape, prior_z.shape)
        loss = KL_loss 
        #print(-det_f , KL_loss )
        #loss = abs(loss)
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(list(self.post_generator.parameters())+list(self.prior_generator.parameters())+list(self.post_net.parameters())+list(self.prior_generator.parameters()), self.clip)
        self.optimizer_G.step()
        #costG = errG_prior - errG_post
        return [('KL_loss', KL_loss.mean().item())]#,('det_f', det_f.mean().item()),('det_g', det_g.sum().item())]
    
    def valid(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.eval()      
        #self.discriminator.eval()
        self.decoder.eval()
        
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        post_z, mu_post, log_s_post, det_f, det_g = self.sample_code_post(x, c)
        prior_z, mu_prior, log_s_prior = self.sample_code_prior(c)
        #errD_post = torch.mean(self.discriminator(torch.cat((post_z, c),1)))
        #errD_prior = torch.mean(self.discriminator(torch.cat((prior_z, c),1)))
        KL_loss = torch.sum(log_s_prior - log_s_post + (torch.exp(log_s_post) + (mu_post)**2)/torch.exp(log_s_prior),1) / 2
        #KL_loss = log_Normal_diag(post_z, mu_post, log_s_post) - log_Normal_diag(prior_z, mu_prior, log_s_prior)
        #KL_loss2 = torch.sum((prior_z - mu_post)**2 / (2 * torch.exp(log_s_post)),1)
        loss =  KL_loss  # -det_f 
        costG = loss.sum()
        dec_target = response[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)
        output = self.decoder(torch.cat((post_z, c),1), None, response[:,:-1], (res_lens-1)) 
        flattened_output = output.view(-1, self.vocab_size) 
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        lossAE = self.criterion_ce(masked_output/self.temp, masked_target)
        return [('valid_loss_AE', lossAE.item()),('valid_loss_G', costG.item())]
    
    def sample(self, context, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):    
        self.context_encoder.eval()
        self.decoder.eval()
        
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        c_repeated = c.expand(repeat, -1)
        prior_z, _ = self.sample_prior(c_repeated)    
        sample_words, sample_lens= self.decoder.sampling(torch.cat((prior_z,c_repeated),1), 
                                                         None, self.maxlen, SOS_tok, EOS_tok, "greedy") 
        return sample_words, sample_lens
    def gen(self, context, prior_z, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):
        self.context_encoder.eval()
        self.decoder.eval()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        c_repeated = c.expand(repeat, -1)
        sample_words, sample_lens= self.decoder.sampling(torch.cat((prior_z,c_repeated),1), 
                                                         None, self.maxlen, SOS_tok, EOS_tok, "greedy")
        return sample_words ,sample_lens
    def sample_latent(self, context, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):
        self.context_encoder.eval()
        #self.decoder.eval()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        c_repeated = c.expand(repeat, -1)
        e,_,_ = self.sample_code_prior(c_repeated)
        prior_z, _ , _ = self.prior_generator((e,0, c_repeated))
        return prior_z ,e
    def sample_latent_post(self, context, context_lens, utt_lens, floors, response, res_lens,repeat):
        self.context_encoder.eval()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        c_repeated = c.expand(repeat, -1)
        x_repeated = x.expand(repeat, -1)
        z_post, z, mu_post, log_s_post, det_f, det_g = self.sample_post(x_repeated, c_repeated)
        return z_post,z
    def adjust_lr(self):
        self.lr_scheduler_AE.step()
    


