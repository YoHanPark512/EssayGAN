# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : sentigan_instructor.py
# @Time         : Created at 2019-07-09
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.optim as optim
import torch.nn as nn
import json

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.EssayGAN_D import EssayGAN_D, EssayGAN_C
from models.EssayGAN_G import EssayGAN_G
from utils import rollout
from utils.cat_data_loader import CatClasDataIter
from utils.data_loader import DisDataIter
from utils.data_loader import GenDataIter
from utils.text_process import tensor_to_tokens, write_tokens

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def write_sents(filename, generator, idx2word_dict, sent_dict, original_sents, data_num, batch_size):
    """Write word tokens to a local file (For Real data)"""
    result = []
    num_sent = []
    source_esssays = []
    generate_samples = {}
    unique = {}
    same = 0
    one_source = 0
    multi_source = 0
    while(len(result) < data_num * 3):
        samples = generator.sample(data_num*4, batch_size)
        essays = tensor_to_tokens(samples, idx2word_dict)
        for essay in essays:
            source = set()
            is_same = False
            if(original_sents.get(tuple(essay), False)):
                same += 1
                continue
            if(unique.get(tuple(essay), False)):
                continue
            ############################
            ##original
            ref = []
            for sent in essay:
                tmp = ["essay_id : {:5},  order : {:3},  score: {:2}".format(essay_id, idx, score) for essay_id, idx, score in sent_dict[sent]]
                ref.append(tmp)
            ############################
                if(len(sent_dict[sent]) == 1 and not is_same):
                    source.add(sent_dict[sent][0][0])

            result.append({"essay": essay,
                           "ref": ref})

            unique[tuple(essay)] = True
            generate_samples[tuple(essay)] = True
            num_sent.append(len(essay))

            if(len(source) != 1):
                source_esssays.append(len(source))
                multi_source += 1
            else:
                one_source += 1

    write = {"same/one source/multi source" : "{}-{}-{}".format(same, one_source, multi_source),
            "avg_sent":sum(num_sent)/len(num_sent),
            "avg_source":sum(source_esssays)/max(len(source_esssays), 1),
            "essays":result}

    with open(filename, 'w') as fout:
        json.dump(write, fout, indent=4)

class EssayGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(EssayGANInstructor, self).__init__(opt)

        self.opt = opt

        # generator, discriminator
        self.gen_list = [EssayGAN_G(opt.gen_embed_dim, opt.gen_hidden_dim, opt.vocab_size, opt.max_seq_len,
                                    cfg.padding_idx, gpu=cfg.CUDA, loss_type=opt.avd_loss_type) for _ in range(cfg.k_label)]
        self.dis = EssayGAN_D(opt.k_label, opt.dis_embed_dim, opt.vocab_size, opt.dis_hidden_dim, opt.dis_hidden_dim*2,
                              opt.max_seq_len, cfg.padding_idx, gpu=cfg.CUDA)
        self.clas = EssayGAN_C(opt.k_label, opt.dis_embed_dim, opt.max_seq_len, cfg.num_rep, opt.extend_vocab_size,
                               cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model(opt)

        self.init_essay_dict()

        self.train_data_list = [GenDataIter(opt.cat_train_data.format(i), drop_last=False) for i in range(cfg.k_label)]
        self.test_data_list = [GenDataIter(opt.cat_train_data.format(i), drop_last=False) for i in range(cfg.k_label)]
        self.train_samples_list = [self.train_data_list[i].target for i in range(cfg.k_label)]

        self.data_num = [len(train_data.tokens) for train_data in self.train_data_list]

        self.original_sents = {tuple(l):True for train_data in self.train_data_list for l in train_data.tokens}

        # Optimizer
        self.gen_opt_list = [optim.Adam(gen.parameters(), lr=cfg.gen_lr) for gen in self.gen_list]
        self.gen_adv_opt_list = [optim.Adam(gen.parameters(), lr=cfg.gen_adv_lr) for gen in self.gen_list]
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.clas_opt = optim.Adam(self.clas.parameters(), lr=cfg.clas_lr)

        # Metrics
        # self.all_metrics.append(self.clas_acc)
        self.all_metrics = [self.nll_gen, self.nll_div]

    def init_model(self, opt):
        if "load_embedding_path" in opt:
            for i in range(cfg.k_label):
                self.log.info('Load embedding generator gen: {}/ from {}'.format(i, opt.load_embedding_path))
                load_embedding = torch.load(opt.load_embedding_path)
                self.gen_list[i].embeddings = nn.Embedding.from_pretrained(load_embedding, padding_idx=cfg.padding_idx, freeze=False)
            self.log.info('Load embedding discriminator: from {}'.format(opt.load_embedding_path))
            load_embedding = torch.load(opt.load_embedding_path)
            self.dis.embeddings = nn.Embedding.from_pretrained(load_embedding, padding_idx=cfg.padding_idx, freeze=False)

        if cfg.dis_pretrain:
            self.log.info(
                'Load pretrained discriminator: {}'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path, map_location='cuda:{}'.format(cfg.device)))
        if cfg.gen_pretrain:
            for i in range(cfg.k_label):
                self.log.info('Load MLE pretrained generator gen: {}'.format(cfg.pretrained_gen_path + '%d' % i))
                self.gen_list[i].load_state_dict(
                    torch.load(cfg.pretrained_gen_path + '%d' % i, map_location='cuda:{}'.format(cfg.device)))
        if cfg.clas_pretrain:
            self.log.info('Load  pretrained classifier: {}'.format(cfg.pretrained_clas_path))
            self.clas.load_state_dict(torch.load(cfg.pretrained_clas_path, map_location='cuda:%d' % cfg.device))

        if cfg.CUDA:
            for i in range(cfg.k_label):
                self.gen_list[i] = self.gen_list[i].cuda()
            self.dis = self.dis.cuda()
            self.clas = self.clas.cuda()

    def init_essay_dict(self):
        with open("./dataset/nltk_sent_tokenizer/ASAP_normalize_essay.prompt{}.json".format(self.opt.prompt)) as f:
            original_essay = json.load(f)

        self.sent_dict = {"[START]":[["start",0,0]]}
        for dic in original_essay:
            score = dic["score"]
            essay_id = dic["essay_id"]
            for idx, sent in enumerate(dic["essay"]):
                sent = sent.replace("  ", " ")
                self.sent_dict.setdefault(sent, [])
                self.sent_dict[sent].append([essay_id, idx, score])

    def _run(self):
        # ===Pre-train Classifier with real data===
        if cfg.use_clas_acc:
            self.log.info('Start training Classifier...')
            self.train_classifier(cfg.PRE_clas_epoch)

        # ===PRE-TRAIN GENERATOR===
        if not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                for i in range(cfg.k_label):
                    torch.save(self.gen_list[i].state_dict(), cfg.pretrained_gen_path + '%d' % i)
                    print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path + '%d' % i))

        self.self_bleu.if_use = True
        self.all_metrics.append(self.self_bleu)

        if(self.opt.mle_in_adv):
            for gen_op in self.gen_opt_list:
                for g in gen_op.param_groups:
                    g["lr"] = 1e-5

        # ===TRAIN DISCRIMINATOR====
        if not cfg.dis_pretrain:
            self.log.info('Starting Discriminator Training...')
            self.train_discriminator(cfg.d_step, cfg.d_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s', self.comb_metrics(fmt_str=True))

        for adv_epoch in range(cfg.ADV_train_epoch):
            self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                if (self.opt.mle_in_adv):
                    self.retrain_generator()
                self.adv_train_generator(cfg.ADV_g_step)  # Generator
                self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')  # Discriminator

                if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                break

    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if(epoch == epochs-1):
                self.self_bleu.if_use = True
                self.all_metrics.append(self.self_bleu)

            if self.sig.pre_sig:
                for i in range(cfg.k_label):
                    pre_loss = self.train_gen_epoch(self.gen_list[i], self.train_data_list[i].loader,
                                                    self.mle_criterion, self.gen_opt_list[i])

                    # ===Test===
                    if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                        if i == cfg.k_label - 1:
                            self.log.info('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (
                                epoch, pre_loss, self.comb_metrics(fmt_str=True)))
                            if cfg.if_save and not cfg.if_test:
                                self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

    def retrain_generator(self):
        """
        Max Likelihood re-training for the generator
        """
        self.log.info("retrain mle")
        for i in range(cfg.k_label):
            pre_loss = self.train_gen_epoch(self.gen_list[i], self.train_data_list[i].loader,
                                            self.mle_criterion, self.gen_opt_list[i])



    def adv_train_generator(self, g_step):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        for i in range(cfg.k_label):
            rollout_func = rollout.ROLLOUT(self.gen_list[i], cfg.CUDA)
            total_g_loss = 0
            for step in range(g_step):
                inp, target = GenDataIter.prepare(self.gen_list[i].sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)

                # ===Train===
                # rewards = rollout_func.get_reward(target, cfg.rollout_num, self.dis, current_k=i)
                rewards = rollout_func.get_reward(target, cfg.rollout_num, self.dis, current_k=0)
                adv_loss = self.gen_list[i].batchPGLoss(inp, target, rewards)
                self.optimize(self.gen_adv_opt_list[i], adv_loss)
                total_g_loss += adv_loss.item()

        # ===Test===
        self.log.info('[ADV-GEN]: %s', self.comb_metrics(fmt_str=True))

    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        global d_loss, train_acc

        for step in range(d_step):
            # prepare loader for training
            real_samples = []
            fake_samples = []
            for i in range(cfg.k_label):
                real_samples.append(self.train_samples_list[i])
                fake_samples.append(self.gen_list[i].sample(cfg.samples_num // cfg.k_label, 8 * cfg.batch_size))

            dis_samples_list = [torch.cat(fake_samples, dim=0)] + real_samples
            # dis_data = CatClasDataIter(dis_samples_list) # sentigan discriminator(k = score +1)
            dis_data = DisDataIter(torch.cat(real_samples, dim=0), torch.cat(fake_samples, dim=0)) # seqgan discriminator(k = 2)

            for epoch in range(d_epoch):
                # ===Train===
                d_loss, train_acc = self.train_dis_epoch(self.dis, dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # ===Test===
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f' % (
                phase, step, d_loss, train_acc))

            if cfg.if_save and not cfg.if_test and phase == 'MLE':
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)

    def cal_metrics_with_label(self, label_i):
        assert type(label_i) == int, 'missing label'

        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen_list[label_i].sample(cfg.samples_num, 8 * cfg.batch_size)
            gen_data = GenDataIter(eval_samples)
            gen_tokens = [" ".join(map(str, remove_values_from_list(tmp.tolist(), 0))) for tmp in eval_samples]
            gen_tokens_s = [" ".join(map(str, remove_values_from_list(tmp.tolist(), 0))) for tmp in self.gen_list[label_i].sample(200, 200)]

            # Reset metrics
            self.nll_gen.reset(self.gen_list[label_i], self.train_data_list[label_i].loader)
            self.nll_div.reset(self.gen_list[label_i], gen_data.loader)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)

        return [metric.get_score() for metric in self.all_metrics]

    def _save(self, phase, epoch):
        """Save model state dict and generator's samples"""
        for i in range(cfg.k_label):
            if phase != 'ADV' or "cross_validation_0" in cfg.save_samples_root:
                torch.save(self.gen_list[i].state_dict(),
                           cfg.save_model_root + 'gen{}_{}_{:05d}.pt'.format(i, phase, epoch))
            save_sample_path = cfg.save_samples_root + 'samples_d{}_{}_{:05d}.data'.format(i, phase, epoch)
            # samples = self.gen_list[i].sample(self.data_num[i] * 4, cfg.batch_size)
            # write_sents(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict), self.sent_dict, self.original_sents)
            write_sents(save_sample_path, self.gen_list[i], self.idx2word_dict, self.sent_dict,
                        self.original_sents, self.data_num[i], cfg.batch_size)
