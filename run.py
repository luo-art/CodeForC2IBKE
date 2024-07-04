import csv
import torch
from helper import *
from dataloader import *
# sys.path.append('./')
from model.models import *
from regularizers import *

class Runner(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits

        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent		= len(self.ent2id)
        self.p.num_rel		= len(self.rel2id)  // 2
        self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)
        self.pos_hr = ddict(list)
        appear_list = np.zeros(40943)
        # f_id_01 = open('0-10.txt', 'w')
        # f_id_12 = open('10-20.txt', 'w')
        # f_id_23 = open('20-30.txt', 'w')
        # f_id_34 = open('30-40.txt', 'w')
        # f_id_45 = open('40-50.txt', 'w')
        # f_id_51 = open('50-100.txt', 'w')
        # f_id_100 = open('101.txt', 'w')

        for split in ['train', 'test', 'valid']:
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))
                self.pos_hr[obj].append([sub,rel])
                self.pos_hr[sub].append([obj,rel + self.p.num_rel])
                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel+self.p.num_rel)].add(sub)
                if split == 'test':
                    appear_list[sub] += 1
                    appear_list[obj] += 1
        ###############################################################
        # for line in open('./data/{}/{}.txt'.format(self.p.dataset, 'test')):
        #     sub, rel, obj = map(str.lower, line.strip().split('\t'))
        #     sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
        #     if appear_list[obj] >= 0 and appear_list[obj] <= 10:
        #         f_id_01.write('{}'.format(sub) + '\t' + '{}'.format(rel) + '\t' + '{}'.format(obj) + '\n')
        #     elif appear_list[obj] >= 10 and appear_list[obj] <= 20:
        #         f_id_12.write('{}'.format(sub) + '\t' + '{}'.format(rel) + '\t' + '{}'.format(obj) + '\n')
        #     elif appear_list[obj] >= 20 and appear_list[obj] <= 30:
        #         f_id_23.write('{}'.format(sub) + '\t' + '{}'.format(rel) + '\t' + '{}'.format(obj) + '\n')
        #     elif appear_list[obj] >= 30 and appear_list[obj] <= 40:
        #         f_id_34.write('{}'.format(sub) + '\t' + '{}'.format(rel) + '\t' + '{}'.format(obj) + '\n')
        #     elif appear_list[obj] >= 40 and appear_list[obj] <= 50:
        #         f_id_45.write('{}'.format(sub) + '\t' + '{}'.format(rel) + '\t' + '{}'.format(obj) + '\n')
        #     elif appear_list[obj] >= 50 and appear_list[obj] <= 100:
        #         f_id_51.write('{}'.format(sub) + '\t' + '{}'.format(rel) + '\t' + '{}'.format(obj) + '\n')
        #     else:
        #         f_id_100.write('{}'.format(sub) + '\t' + '{}'.format(rel) + '\t' + '{}'.format(obj) + '\n')
        ###############################################################
        weight = appear_list / np.max(appear_list) * 0.9 + 0.1
        self.weight = torch.Tensor(weight).cuda()
        self.data = dict(self.data)
        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+self.p.num_rel)].add(sub)
        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples  = ddict(list)

        for sub, rel, obj in self.data['train']:
            rel_inv = rel + self.p.num_rel  # inverse_rel相反关系
            self.triples['train'].append(
                {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
            self.triples['train'].append(
                {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return  DataLoader(
                    dataset_class(self.triples[split], self.p),
                    batch_size      = batch_size,
                    shuffle         = shuffle,
                    num_workers     = max(0, self.p.num_workers),
                    collate_fn      = dataset_class.collate_fn
                )

        self.data_iter = {
            'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
            'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
            'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
            'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
            'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []
        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)
            edge_index.append((obj,sub))
            edge_type.append(rel+self.p.num_rel)
        for item in range(self.p.num_ent):
            edge_index.append((torch.tensor(item),torch.tensor(item)))
        edge_index	= torch.LongTensor(edge_index).to(self.device).t()
        edge_type	= torch.LongTensor(edge_type). to(self.device)


        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p			= params
        self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model        = self.add_model(self.p.model)
        self.optimizer    = self.add_optimizer(self.model.parameters())


    def add_model(self, model):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """
        model_name = '{}'.format(model)
        regularizer = DURA_RESCAL(args.reg)
        model = MCIK(self.edge_index, self.edge_type,self, params=self.p)
        regularizer = [regularizer, N3(args.reg)]
        self.regularizer = regularizer[0]
        model.to(self.device)
        for reg in regularizer:  # regularizer=[DURA_RESCAL(),N3()]
            reg.to(self.device)
        self.regularizer = regularizer[0]
        return model

    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        return torch.optim.Adagrad(parameters, lr=self.p.lr)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        Head, Relation, Tails, labels
        """
        if split == 'train':
            triple, label,neg_tail = [ _.to(self.device) for _ in batch]
            return triple,triple[:, 0], triple[:, 1], triple[:, 2], label,neg_tail
        else:
            triple, label = [ _.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict'	: self.model.state_dict(),
            'best_test'	: self.best_test,
            'best_epoch'	: self.best_epoch,
            'optimizer'	: self.optimizer.state_dict(),
            'args'		: vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state			= torch.load(load_path)
        state_dict		= state['state_dict']
        self.best_test		= state['best_test']
        self.best_test_mrr	= self.best_test['mrr']
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        left_results  = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results       = get_combined_results(left_results, right_results)
        self.logger.info('[Epoch {} {}]: MRR: Tail : {:.3}, Head : {:.3}, Avg : {:.3},Hits@[1,3,10]:[{:.3},{:.3},{:.3}],MR:{:.3}'.
                         format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr'],results['hits@1'],results['hits@3'],results['hits@10'],results['mr']))
        return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """

        # if split == 'test':
        #     self.model.eval()
        #     with torch.no_grad():
        #         results = {}
        #         train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])
        #     # f_rank_all = open(
        #     #     './ranks_index_top10/{}_{}_{}_{}_ranks_{}.csv'.format(self.p.dataset, mode.split('_')[0],
        #     #                                                           self.p.model, self.p.score_func,
        #     #                                                           self.p.name), 'w', encoding="UTF8",
        #     #     newline='')
        #     # writer_all = csv.writer(f_rank_all, delimiter=",")
        #     #
        #     # # 保存三元组针对当前尾实体的排名情况
        #     # f_rank_index = open(
        #     #     './ranks/{}_{}_{}_{}_ranks_{}.txt'.format(self.p.dataset, mode.split('_')[0], self.p.model,
        #     #                                               self.p.score_func, self.p.name), 'w', encoding="UTF8",
        #     #     newline='')
        #     # writer = csv.writer(f_rank_index, delimiter='\t')
        #     for step, batch in enumerate(train_iter):
        #         sub, rel, obj, label = self.read_batch(batch, split)
        #         pred, _, _, _ = self.model.forward(sub, rel, obj, None, None)
        #     #
        #     #     ####针对所有排名 返回排名前10的索引 前三列对应h r t后10列对应排名前10的索引
        #     #     rank_index = torch.argsort(pred, dim=1, descending=True)[:, :10]
        #     #     t_rank_index = torch.cat([sub.unsqueeze(1), rel.unsqueeze(1), obj.unsqueeze(1), rank_index], dim=1)
        #     #     t_rank_index = t_rank_index.cpu().numpy().astype(int)
        #     #     writer_all.writerows(t_rank_index)
        #     #     ######
        #     #
        #     #     b_range = torch.arange(pred.size()[0], device=self.device)
        #     #     target_pred = pred[b_range, obj]
        #     #     pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
        #     #     pred[b_range, obj] = target_pred
        #     #     ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
        #     #         b_range, obj]
        #     #     ranks = ranks.float()
        #     #
        #     #     #####三元组排名
        #     #     triple_rank = torch.cat([sub.unsqueeze(1), rel.unsqueeze(1), obj.unsqueeze(1), ranks.unsqueeze(1)],
        #     #                             dim=1)
        #     #     triple_rank = triple_rank.cpu().numpy().astype(int)
        #     #     writer.writerows(triple_rank)
        #         results['count'] = torch.numel(ranks) + results.get('count', 0.0)
        #         results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
        #         results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
        #         for k in range(10):
        #             results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
        #                 'hits@{}'.format(k + 1), 0.0)
        #         if step % 100 == 0:
        #             self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))
        #     return results

        #############################################################
        self.model.eval()
        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label	= self.read_batch(batch, split)
                pred,_,_,_			= self.model.forward(sub, rel,obj,None,None,None,0)
                b_range			= torch.arange(pred.size()[0], device=self.device)
                target_pred		= pred[b_range, obj]
                pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] 	= target_pred
            ##############################################################################
                ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
                ranks 			= ranks.float()
                results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
                results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
                results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
                for k in range(10):
                    results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        return results


    def run_epoch(self, epoch, val_mrr = 0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            tripe,sub, rel, obj, label,neg_tail = self.read_batch(batch, 'train')
            p_hr = []
            for i in tripe:
                hr_sample = self.pos_hr[i[2].item()]
                random.shuffle(hr_sample)  # random.shuffle()用于将一个列表中的元素打乱顺序
                p_hr.append(hr_sample[0])
            p_hr = torch.tensor(p_hr)
            p_hr = p_hr.cuda()
            pred,cl_loss,factors,score	= self.model.forward(sub, rel,obj,p_hr,neg_tail,'train',1)
            l_reg = self.regularizer.forward(factors)
            # loss = self.model.loss(tmp,obj)
            #self.weight的维度是【40943】
            weight = torch.stack([self.weight.index_select(0, neg_tail[i]) for i in range(score.shape[0])], 0)
            loss = 0
            for i in range(score.shape[0]):
                loss1 = torch.nn.CrossEntropyLoss(reduction="mean",weight = weight[i])
                loss += loss1(score[i],torch.tensor(0).cuda())
                #这里score的维度是【128 * 41】，obj是【128】
            loss /= score.shape[0]
            loss2 = loss  + l_reg + cl_loss
            loss2.backward()
            self.optimizer.step()
            losses.append(loss2.item())

            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Test MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_test_mrr, self.p.name))

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss


    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_test_mrr, self.best_test, self.best_epoch, test_mrr = 0., {}, 0, 0.
        save_path = os.path.join('./checkpoints', self.p.name)
        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')
        # if self.p.dataset == 'FB15k-237' or self.p.dataset == 'WN18RR':
        #     f = open('./results_nn/nn_{}_{}_{}_{}'.format(self.p.dataset, self.p.model, self.p.score_func,
        #                                                   self.p.name), 'w')
        #     f.write('all_{}'.format(self.p.dataset) + '\t' + 'tail' + '\t' + 'head' + '\t' + 'mean' + '\n')
        #     rstt = {}
        #
        #     for m in ['tail', 'head']:
        #         f_rank = open('./ranks/{}_{}_{}_{}_ranks_{}.txt'.format(self.p.dataset, m, self.p.model,
        #                                                                 self.p.score_func, 'testrun'),
        #                       'r').read().split('\n')
        #         f_rank.remove(f_rank[-1])
        #         print('len(f_rank):{}'.format(len(f_rank)))
        #         if m == 'head':
        #             mr = 0
        #             mrr = 0
        #             hit1 = 0
        #             hit3 = 0
        #             hit10 = 0
        #             for line in f_rank:
        #                 s = line.split('\t')
        #                 mr += int(s[3])
        #                 mrr += 1 / int(s[3])
        #                 if int(s[3]) <= 1:
        #                     hit1 += 1
        #                 if int(s[3]) <= 3:
        #                     hit3 += 1
        #                 if int(s[3]) <= 10:
        #                     hit10 += 1
        #             mr = mr / len(f_rank)
        #             mrr = mrr / len(f_rank)
        #             hit1 = hit1 / len(f_rank)
        #             hit3 = hit3 / len(f_rank)
        #             hit10 = hit10 / len(f_rank)
        #             rstt['head_mr'] = mr
        #             rstt['head_mrr'] = mrr
        #             rstt['head_hit1'] = hit1
        #             rstt['head_hit3'] = hit3
        #             rstt['head_hit10'] = hit10
        #         if m == 'tail':
        #             mr = 0
        #             mrr = 0
        #             hit1 = 0
        #             hit3 = 0
        #             hit10 = 0
        #             for line in f_rank:
        #                 s = line.split('\t')
        #                 mr += int(s[3])
        #                 mrr += 1 / int(s[3])
        #                 if int(s[3]) <= 1:
        #                     hit1 += 1
        #                 if int(s[3]) <= 3:
        #                     hit3 += 1
        #                 if int(s[3]) <= 10:
        #                     hit10 += 1
        #             mr = mr / len(f_rank)
        #             mrr = mrr / len(f_rank)
        #             hit1 = hit1 / len(f_rank)
        #             hit3 = hit3 / len(f_rank)
        #             hit10 = hit10 / len(f_rank)
        #             rstt['tail_mr'] = mr
        #             rstt['tail_mrr'] = mrr
        #             rstt['tail_hit1'] = hit1
        #             rstt['tail_hit3'] = hit3
        #             rstt['tail_hit10'] = hit10
        #     rstt['mr'] = (rstt['tail_mr'] + rstt['head_mr']) / 2
        #     rstt['mrr'] = (rstt['tail_mrr'] + rstt['head_mrr']) / 2
        #     rstt['hit1'] = (rstt['tail_hit1'] + rstt['head_hit1']) / 2
        #     rstt['hit3'] = (rstt['tail_hit3'] + rstt['head_hit3']) / 2
        #     rstt['hit10'] = (rstt['tail_hit10'] + rstt['head_hit10']) / 2
        #
        #     f.write('mr' + '\t' + str(rstt['tail_mr']) + '\t' + str(rstt['head_mr']) + '\t' + str(rstt['mr']) + '\n')
        #     f.write(
        #         'mrr' + '\t' + str(rstt['tail_mrr']) + '\t' + str(rstt['head_mrr']) + '\t' + str(rstt['mrr']) + '\n')
        #     f.write('hit1' + '\t' + str(rstt['tail_hit1']) + '\t' + str(rstt['head_hit1']) + '\t' + str(
        #         rstt['hit1']) + '\n')
        #     f.write('hit3' + '\t' + str(rstt['tail_hit3']) + '\t' + str(rstt['head_hit3']) + '\t' + str(
        #         rstt['hit3']) + '\n')
        #     f.write('hit10' + '\t' + str(rstt['tail_hit10']) + '\t' + str(rstt['head_hit10']) + '\t' + str(
        #         rstt['hit10']) + '\n')
        #     f.write('\n')
        #
        #
        #     for tri in ['1-1', '1-n', 'n-1', 'n-n']:
        #         f.write('{}_{}'.format(tri, self.p.dataset) + '\t' + 'tail' + '\t' + 'head' + '\t' + 'mean' + '\n')
        #         f_tri = open('./data/{}/{}.txt'.format(self.p.dataset, tri), 'r').read().split('\n')
        #         f_tri.remove(f_tri[-1])
        #         rst = {}
        #         print('len(f_tri):{}_{}'.format(tri, len(f_tri)))
        #         for m in ['tail', 'head']:
        #             f_rank = open('./ranks/{}_{}_{}_{}_ranks_{}.txt'.format(self.p.dataset, m, self.p.model,
        #                                                                     self.p.score_func, 'testrun'),
        #                           'r').read().split('\n')
        #             f_rank.remove(f_rank[-1])
        #             tri_rank = {}
        #
        #             for line in f_rank:
        #                 s = line.split('\t')
        #                 tri_rank[s[0] + '\t' + s[1] + '\t' + s[2]] = s[3]
        #
        #             if m == 'head':
        #                 mr = 0
        #                 mrr = 0
        #                 hit1 = 0
        #                 hit3 = 0
        #                 hit10 = 0
        #                 for line in f_tri:
        #                     s = line.split('\t')
        #                     newline = s[2] + '\t' + '{}'.format(str(int(s[1]) + self.p.num_rel)) + '\t' + s[0]
        #                     mr += int(tri_rank[newline])
        #                     mrr += (1 / int(tri_rank[newline]))
        #                     if int(tri_rank[newline]) <= 1:
        #                         hit1 += 1
        #                     if int(tri_rank[newline]) <= 3:
        #                         hit3 += 1
        #                     if int(tri_rank[newline]) <= 10:
        #                         hit10 += 1
        #                 mr = mr / len(f_tri)
        #                 mrr = mrr / len(f_tri)
        #                 hit1 = hit1 / len(f_tri)
        #                 hit3 = hit3 / len(f_tri)
        #                 hit10 = hit10 / len(f_tri)
        #                 rst['head_mr'] = mr
        #                 rst['head_mrr'] = mrr
        #                 rst['head_hit1'] = hit1
        #                 rst['head_hit3'] = hit3
        #                 rst['head_hit10'] = hit10
        #
        #             if m == 'tail':
        #                 mr = 0
        #                 mrr = 0
        #                 hit1 = 0
        #                 hit3 = 0
        #                 hit10 = 0
        #                 for line in f_tri:
        #                     mr += int(tri_rank[line])
        #                     mrr += (1 / int(tri_rank[line]))
        #                     if int(tri_rank[line]) <= 1:
        #                         hit1 += 1
        #                     if int(tri_rank[line]) <= 3:
        #                         hit3 += 1
        #                     if int(tri_rank[line]) <= 10:
        #                         hit10 += 1
        #                 mr = mr / len(f_tri)
        #                 mrr = mrr / len(f_tri)
        #                 hit1 = hit1 / len(f_tri)
        #                 hit3 = hit3 / len(f_tri)
        #                 hit10 = hit10 / len(f_tri)
        #                 rst['tail_mr'] = mr
        #                 rst['tail_mrr'] = mrr
        #                 rst['tail_hit1'] = hit1
        #                 rst['tail_hit3'] = hit3
        #                 rst['tail_hit10'] = hit10
        #
        #         rst['mr'] = (rst['tail_mr'] + rst['head_mr']) / 2
        #         rst['mrr'] = (rst['tail_mrr'] + rst['head_mrr']) / 2
        #         rst['hit1'] = (rst['tail_hit1'] + rst['head_hit1']) / 2
        #         rst['hit3'] = (rst['tail_hit3'] + rst['head_hit3']) / 2
        #         rst['hit10'] = (rst['tail_hit10'] + rst['head_hit10']) / 2
        #         f.write('mr' + '\t' + str(rst['tail_mr']) + '\t' + str(rst['head_mr']) + '\t' + str(rst['mr']) + '\n')
        #         f.write(
        #             'mrr' + '\t' + str(rst['tail_mrr']) + '\t' + str(rst['head_mrr']) + '\t' + str(rst['mrr']) + '\n')
        #         f.write('hit1' + '\t' + str(rst['tail_hit1']) + '\t' + str(rst['head_hit1']) + '\t' + str(
        #             rst['hit1']) + '\n')
        #         f.write('hit3' + '\t' + str(rst['tail_hit3']) + '\t' + str(rst['head_hit3']) + '\t' + str(
        #             rst['hit3']) + '\n')
        #         f.write('hit10' + '\t' + str(rst['tail_hit10']) + '\t' + str(rst['head_hit10']) + '\t' + str(
        #             rst['hit10']) + '\n')
        #         f.write('\n')
        #     f.close()
        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss  = self.run_epoch(epoch, test_mrr)
            val_results = self.evaluate('valid', epoch)
            test_results = self.evaluate("test",epoch)

            if test_results['mrr'] > self.best_test_mrr:
                self.best_test	   = test_results
                self.best_test_mrr  = test_results['mrr']
                self.best_epoch	   = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                if kill_cnt > 30:
                    self.logger.info("Early Stopping!!")
                    break
            self.logger.info('[Epoch {}]: Training Loss: {:.5}, Test MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_test_mrr))
        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        test_results = self.evaluate('test', epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-name', default='testrun_' + time.strftime("[%Y-%m-%d__%H-%M-%S]", time.localtime()), help='Name of the experiment')
    parser.add_argument('-score_func', default='similarty', help='caculate the score of entities')
    parser.add_argument('-data',		dest='dataset',         default="WN18RR", help='Dataset to use, default: WN18RR ')
    parser.add_argument('-model',		dest='model',		default='C2IBKE',		help='Model Name')
    parser.add_argument('-regularizer', type=str, default='DURA_RESCAL')
    parser.add_argument('-reg', default=0.1, type=float,help="Regularization weight")
    parser.add_argument('-lamda', dest='lamda', default=0.02, type=float, help='the paramater of ib_loss')
    parser.add_argument('-beta', dest='beta', default=1, type=float, help='the parasmater of node_cl_loss')
    parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
    parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
    parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=200,  	help='Number of epochs')
    parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',		type=float,             default=0.1,			help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
    parser.add_argument('-num_workers',	type=int,               default=0,                     help='Number of processes to construct batches')
    parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')
    parser.add_argument("--neg_sampe_ratio", dest='neg_sampe_ratio', default=10000, type=int, help='neg_sampe_ratio')
    parser.add_argument("--seed", dest='seed', default=1314, type=int, help='Seed for randomization')
    parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
    parser.add_argument('-embed_dim',	dest='embed_dim', 	default=200,   type=int, 	help='Embedding dimension to give as input to score function')
    parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
    parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
    args = parser.parse_args()
    set_gpu(args.gpu)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(args.seed)  # 为CPU设置随机种子
    model = Runner(args)
    model.fit()
