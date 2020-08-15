from .regimports import *
from fastprogress import master_bar,progress_bar
import pdb
class CancelTrainException(Exception):
    pass
class CancelBatchException(Exception):
    pass
class TrainPacket:
    def __init__(self,model,opt,loss_func,dls):
        self.model = model
        self.opt = opt
        self.loss_func =loss_func
        self.dls = dls 

class CallBackHandler:
    def __init__(self,cbs):
        self.cbs = cbs
    def __call__(self,name,runner):
        for c in sorted(self.cbs,key=lambda x: x._order):
            f = getattr(c,name,None)
            if f is not None:
                f(runner)

class Runner:
    def __init__(self,trainpacket,cbh):
        self.trainpacket = trainpacket
        self.cbh = cbh
    @property
    def model(self):
        return self.trainpacket.model
    @property
    def opt(self):
        return self.trainpacket.opt
    @property
    def loss_func(self):
        return self.trainpacket.loss_func
    @property
    def dls(self):
        return self.trainpacket.dls
    
    def one_batch(self,batch):
        try:
            self.batch = batch
            self('begin_batch')
            self.preds = self.model(*self.xb)
            self('after_preds')
            if self.mask is not None:
                self.preds = self.preds[self.mask]
                self.yb = self.yb[self.mask]
            self.loss = self.loss_func(self.preds,self.yb)
            if self.in_train:
                self.loss.backward()
                self('after_backward')
                self.opt.step()
                self('after_step')
                self.opt.zero_grad()
        except CancelBatchException as e:
            print(repr(e))
        finally:
            self('after_batch')

    def all_batches(self,dl,parent=None):
        for batch in progress_bar(dl,parent=parent):
            self.batch = batch
            self.one_batch(batch)
            
    def fit(self,epochs=100):
        try:
            self('begin_fit')
            self.mb = master_bar(range(epochs),total=epochs)
            for epoch in self.mb:
                self.epoch=epoch
                self('begin_epoch')
                self.in_train=True
                self.model.train()
                self.all_batches(self.dls['train'],parent=self.mb)
                self('after_training')
                self.in_train=False
                self('before_validation')
                self.model.eval()
                with torch.no_grad():
                    self.all_batches(self.dls['valid'],parent=self.mb)
                self('after_validation')
                self('after_epoch')
        except CancelTrainException as e:
            print(repr(e))
        finally:
            self('after_train')
    
    def __call__(self,name):
        self.cbh(name,self)
        
class CleanerCallBack:
    _order=10
    def after_batch(self,runner):
        del runner.batch,runner.yb.runner.xb,runner.mask,runner.loss,runner.preds
        gc.collect()
        torch.cuda.empty_cache()
    def __repr__(self):
        return self.__class__.__name__


class TrainRecorderCallBack:
    _order=0
    def __init__(self,metric_funcs=[(accuracy_score,'summable')],resume=False):
        if not isinstance(metric_funcs,list):
            metric_funcs=[metric_funcs]
        self.metric_funcs,self.summables = list(zip(*metric_funcs))
        self.resume=resume

    def begin_fit(self,runner):
        if self.resume:
            return
        self.losses=dict(valid_batch=[],train_batch=[],valid_epoch=[],train_epoch=[])
        self.metrics = {}
        for m in self.metric_funcs:
            for mode in ['train','valid']:
                self.metrics[mode+'_'+m.__class__.__name__]=[]

    def begin_epoch(self,runner):
        self.epoch_measures=dict(train_count=0,valid_count=0,train_loss=0,\
            valid_loss=0,train_preds=[],valid_preds=[],\
                valid_groundtruths=[],test_groundtruths=[])
        for m,summable in zip(self.metric_funcs,self.summables):
            if summable:
                self.epoch_measures['train'+'_'+m.__class__.__name__]=0
                self.epoch_measures['valid'+'_'+m.__class__.__name__]=0

    def after_batch(self,runner):
        if runner.in_train:
            mode = 'train'
        else:
            mode = 'valid'
        self.losses[mode+'_batch'].append(runner.loss)
        self.epoch_measures[mode+'_count']+=runner.yb.shape[0]
        self.epoch_measures[mode+'_loss']+=runner.loss.item()*runner.yb.shape[0]
        for m,summable in zip(self.metric_funcs,self.summables):
            if summable:
                self.epoch_measures[mode+'_'+m.__class__.__name__]+=m(runner.preds.detach().cpu().numpy(),runner.yb.detach().cpu().numpy())*runner.yb.shape[0]
        if all(self.summables):
            return
        self.epoch_measures[mode+'_preds'].append(runner.preds.detach().cpu().numpy())
        self.epoch_measures[mode+'_groundtruths'].append(runner.yb.detach().cpu().numpy())

    def after_epoch(self,runner):
        for mode in ['train','valid']:
            self.losses[mode+'_epoch'].append(self.epoch_measures[mode+'_loss']/self.epoch_measures[mode+'_count'])
            for m,summable in zip(self.metric_funcs,self.summables):
                if summable:
                    self.metrics[mode+'_'+m.__class__.__name__].append(self.epoch_measures[mode+'_'+m.__class__.__name__]/self.epoch_measures[mode+'_count'])
                else:
                    self.metrics[mode+'_'+m.__class__.__name__].append(m(self.epoch_measures[mode+'_preds'],self.epoch_measures[mode+'_groundtruths']))
        train_content=f"Loss:{self.losses['train_epoch'][-1]:.4f} "
        valid_content=f"Loss:{self.losses['valid_epoch'][-1]:.4f} "
        for m in self.metric_funcs:
            train_content+=f'{m.__class__.__name__}: {self.metrics["train"+"_"+m.__class__.__name__][-1]:.4f}'
            valid_content+=f'{m.__class__.__name__}: {self.metrics["valid"+"_"+m.__class__.__name__][-1]:.4f}'
        runner.prints = f"Epoch[{runner.epoch}]: Training Stats--> Loss:{self.losses['train_epoch'][-1]:.4f} {train_content}"
        runner.prints+=f"Validation Stats--> Loss:{self.losses['valid_epoch'][-1]:.4f} {valid_content}"

    def __repr__(self):
        return self.__class__.__name__

class PrintStatsCallBack:
    _order = 2
    def after_epoch(self,runner):
        runner.mb.write(runner.prints)
    def __repr__(self):
        return self.__class__.__name__

class PreProcessingCallBack:
    _order=0
    def __init__(self,xnames=[],mask_name='mask',ynames=['y']):
        self.xnames,self.mask_name,self.ynames = xnames,mask_name,ynames

    def begin_batch(self,runner):
        temp=[]
        for x in self.xnames:
            if x=='graph':
                temp.append(getattr(runner.batch,x))
            else:
                temp.append(getattr(runner.batch,x).cuda())
        runner.xb = temp
        mask = getattr(runner.batch,self.mask_name)
        if mask is not None:
            runner.mask = mask.cuda()
        if len(self.ynames)==1:
            runner.yb = getattr(runner.batch,self.ynames[0]).cuda()
        else:
            temp=[]
            for x in self.ynames:
                temp.append(getattr(runner.batch,y).cuda())
            runner.yb = temp        
        
    def __repr__(self):
        return self.__class__.__name__    

class EarlyStoppingCallBack:
    def __init__(self,patience):
        pass
        
    def __repr__(self):
        return self.__class__.__name__

def one_batch_simple(model,X,batch_node_idxs,target,edge_index,loss_func,opt,is_train=False):
    # moving to gpu
    target = target.cuda()
    # predicting on a batch. xb--> node features, edge_index--> sparse adjacency
    probs = model(X,edge_index)[batch_node_idxs]
    # computing loss
    loss = loss_func(probs, target)
    if is_train:
        loss.backward()
        opt.step()
        opt.zero_grad()
    # from raw predicitons to class 
    preds =  torch.argmax(probs,dim=1)
    acc = preds.eq(target.view_as(preds)).float().mean()
    return {'loss':loss.item(),'accuracy':acc.item()}


def train_simple(model,X,edge_index,y,dls,loss_func,opt):
    batch_losses=[]
    batch_accuracies=[]
    # walk through batch by batch of all training nodes 
    for batch_node_idxs in progress_bar(dls['train']):
        temp = one_batch(model,X,batch_node_idxs,y[batch_node_idxs],edge_index,loss_func,opt,is_train=True)
        batch_losses.append(temp['loss'])
        batch_accuracies.append(temp['accuracy']*len(batch_node_idxs))
    return {'TrainingEpochLosses':batch_losses,'TrainingEpochAccuracies':batch_accuracies}

def test_simple(model,X,edge_index,y,dls,loss_func,opt,key='valid'):
    batch_losses=[]
    batch_accuracies=[]
    # walk through batch by batch of all training nodes
    for batch_node_idxs in progress_bar(dls[key]):
        temp = one_batch_simple(model,X,batch_node_idxs,y[batch_node_idxs],edge_index,loss_func,opt,is_train=False)
        batch_losses.append(temp['loss'])
        batch_accuracies.append(temp['accuracy']*len(batch_node_idxs))
    return {'EpochLosses':batch_losses,'EpochAccuracies':batch_accuracies}


def run(trainpacket,epochs=10):
    ## running training followed by validation
    all_training_losses=[]
    all_validation_losses=[]
    all_training_accuracies=[]
    all_validation_accuracies=[]
    # moving to cuda
    pbar =  tqdm(range(epochs))
    for epoch in pbar:
        train_results=train_simple(model,X,edge_index,y,dls,loss_func,opt)
        test_results=test_simple(model,X,edge_index,y,dls,loss_func,opt)
        ## summarizing losses 
        all_training_losses.append(np.mean(train_results['TrainingEpochLosses']))
        all_validation_losses.append(np.mean(test_results['EpochLosses']))
        all_training_accuracies.append(np.sum(train_results['TrainingEpochAccuracies'])/len(dls['train']))
        all_validation_accuracies.append(np.sum(test_results['EpochAccuracies'])/len(dls['valid']))
        pbar.set_description(f"Epoch --> {epoch}, Stats--> Training Loss:{all_training_losses[-1]:.3f}, Validation Loss:{all_validation_losses[-1]:.3f}, Training accuracy:{all_training_accuracies[-1]:.3f}, Validation Accuracy:{all_validation_accuracies[-1]:.3f}")
    del X,edge_index
    gc.collect()
    torch.cuda.empty_cache()
    return {'TrainingLosses':all_training_losses,'TrainingAccuracy':all_training_accuracies,'validationLosses':all_validation_losses,'ValidationAccuracy':all_validation_accuracies}


