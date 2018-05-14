import argparse
import torch
import codecs
import os
import math

from torch.autograd import Variable
from itertools import count

import onmt.ModelConstructor
import onmt.translate.Beam
import onmt.io
import onmt.opts
import tables


def make_translator(opt, report_score=True, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train_mm.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    # loading checkpoint just to find multimodal model type
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    opt.multimodal_model_type = checkpoint['opt'].multimodal_model_type
    del checkpoint

    if opt.batch_size > 1:
        print("Batch size > 1 not implemented! Falling back to batch_size = 1 ...")
        opt.batch_size = 1

    # load test image features
    test_file = tables.open_file(opt.path_to_test_img_feats, mode='r')
    if opt.multimodal_model_type in ['imgd', 'imge', 'imgw']:
        test_img_feats = test_file.root.global_feats[:]
    elif opt.multimodal_model_type in ['src+img']:
        test_img_feats = test_file.root.local_feats[:]
    else:
        raise Exception("Model type not implemented: %s" % opt.multimodal_model_type)
    test_file.close()

    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam",
                        "data_type", "replace_unk", "gpu", "verbose"]}

    translator = TranslatorMultimodal(model, fields,
                                      global_scorer=scorer,
                                      out_file=out_file,
                                      report_score=report_score,
                                      copy_attn=model_opt.copy_attn,
                                      test_img_feats=test_img_feats,
                                      multimodal_model_type=opt.multimodal_model_type,
                                      **kwargs)
    return translator


class TranslatorMultimodal(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """

    def __init__(self,
                 model,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 sample_rate='16000',
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 data_type="text",
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,
                 test_img_feats=None,
                 multimodal_model_type=None):
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge

        # multimodal
        self.test_img_feats = test_img_feats
        self.multimodal_model_type = multimodal_model_type

        assert (not test_img_feats is None), \
            'Please provide file with test image features.'
        assert (not multimodal_model_type is None), \
            'Please provide the multimodal model type name.'

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self, src_dir, src_path, tgt_path,
                  batch_size, attn_debug=False):
        data = onmt.io.build_dataset(self.fields,
                                     self.data_type,
                                     src_path,
                                     tgt_path,
                                     src_dir=src_dir,
                                     sample_rate=self.sample_rate,
                                     window_size=self.window_size,
                                     window_stride=self.window_stride,
                                     window=self.window,
                                     use_filter_pred=self.use_filter_pred)

        data_iter = onmt.io.OrderedIterator(
            dataset=data, device=self.gpu,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        for sent_idx, batch in enumerate(data_iter):
            batch_data = self.translate_batch(batch, data, sent_idx)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[0]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    os.write(1, output.encode('utf-8'))

                # Debug attention.
                if attn_debug:
                    srcs = trans.src_raw
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *trans.src_raw) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))

        if self.report_score:
            self._report_score('PRED', pred_score_total, pred_words_total)
            if tgt_path is not None:
                self._report_score('GOLD', gold_score_total, gold_words_total)
                if self.report_bleu:
                    self._report_bleu(tgt_path)
                if self.report_rouge:
                    self._report_rouge(tgt_path)

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))
        return all_scores

    def translate_batch(self, batch, data, sent_idx):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           sent_idx: the sentence idxs mapping to the image features


        Todo:
           Shouldn't need the original dataset.
        """

        # load image features for this minibatch into a pytorch Variable
        img_feats = torch.from_numpy(self.test_img_feats[sent_idx])
        img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
        img_feats = img_feats.unsqueeze(0)
        if next(self.model.parameters()).is_cuda:
            img_feats = img_feats.cuda()
        else:
            img_feats = img_feats.cpu()

        # project image features
        img_proj = self.model.encoder_images(img_feats)

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        # enc_states, memory_bank = self.model.encoder(src, src_lengths)
        # dec_states = self.model.decoder.init_decoder_state(
        #     src, memory_bank, enc_states)

        if self.multimodal_model_type == 'imge':
            # create initial hidden state differently for GRU/LSTM
            if self.model._evaluate_is_tuple_hidden(src, src_lengths):
                enc_init_state = (img_proj, img_proj)
            else:
                enc_init_state = img_proj
            # initialise encoder with image features
            enc_states, memory_bank = self.model.encoder(src, src_lengths, enc_init_state)
            # traditional decoder
            dec_states = self.model.decoder.init_decoder_state(src, memory_bank, enc_states)
        elif self.multimodal_model_type == 'imgd':
            # traditional encoder
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            # combine encoder final hidden state with image features
            enc_init_state = self.model._combine_enc_state_img_proj(enc_states, img_proj)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                                            src, memory_bank, enc_init_state)
        elif self.multimodal_model_type == 'imgw':
            # use image features as words in the encoder
            enc_states, memory_bank = self.model.encoder(src, img_feats=img_proj, lengths=src_lengths)
            # update the lengths variable with the new source lengths after incorporating image feats
            src_lengths = self.model.encoder.updated_lengths
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                src, memory_bank, enc_states)
        elif self.multimodal_model_type == 'src+img':
            # traditional encoder
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                src, memory_bank, img_proj, enc_states)
        else:
            raise Exception("Multi-modal model not implemented: %s" % self.multimodal_model_type)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)

        # image features are in (batch x len x feats),
        # but rvar() function expects (len x batch x feats)
        img_proj = rvar(img_proj.transpose(0, 1).data)
        # return it back to (batch x len x feats)
        img_proj = img_proj.transpose(0, 1)

        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            # dec_out, dec_states, attn = self.model.decoder(
            #     inp, memory_bank, dec_states, memory_lengths=memory_lengths)
            if self.multimodal_model_type in ['imgw', 'imge', 'imgd']:
                dec_out, dec_states, attn = self.model.decoder(
                    inp, memory_bank, dec_states, memory_lengths=memory_lengths)
            elif self.multimodal_model_type == 'src+img':
                dec_out, dec_out_imgs, dec_states, attn = self.model.decoder(
                    inp, memory_bank, img_proj, dec_states, memory_lengths=memory_lengths)
            else:
                raise Exception("Multi-modal model type not implemented: %s"%(
                    self.multimodal_model_type))

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data, sent_idx)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data, sent_idx):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        # enc_states, memory_bank = self.model.encoder(src, src_lengths)
        # dec_states = \
        #     self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        # load image features for this minibatch into a pytorch Variable
        img_feats = torch.from_numpy(self.test_img_feats[sent_idx])
        img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
        img_feats = img_feats.unsqueeze(0)
        if next(self.model.parameters()).is_cuda:
            img_feats = img_feats.cuda()
        else:
            img_feats = img_feats.cpu()

        # project image features
        img_proj = self.model.encoder_images(img_feats)
        if self.multimodal_model_type == 'imge':
            # initialise encoder with image features
            enc_states, memory_bank = self.model.encoder(src, src_lengths, img_proj)
            # traditional decoder
            dec_states = self.model.decoder.init_decoder_state(src, memory_bank, enc_states)
        elif self.multimodal_model_type == 'imgd':
            # traditional encoder
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            # combine encoder final hidden state with image features
            enc_init_state = self.model._combine_enc_state_img_proj(enc_states, img_proj)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                src, memory_bank, enc_init_state)
        elif self.multimodal_model_type == 'imgw':
            # use image features as words in the encoder
            enc_states, memory_bank = self.model.encoder(src, img_feats=img_proj, lengths=src_lengths)
            # update the lengths variable with the new source lengths after incorporating image feats
            src_lengths = self.model.encoder.updated_lengths
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(src, memory_bank, enc_states)
        elif self.multimodal_model_type == 'src+img':
            # traditional encoder
            enc_states, memory_bank = self.model.encoder(src, src_lengths)
            # initialise decoder
            dec_states = self.model.decoder.init_decoder_state(
                src, memory_bank, img_proj, enc_states)
        else:
            raise Exception("Multi-modal model not implemented: %s" % self.multimodal_model_type)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        # dec_out, _, _ = self.model.decoder(
        #     tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        if self.multimodal_model_type in ['imgw', 'imge', 'imgd']:
            dec_out, dec_states, attn = self.model.decoder(
                tgt_in, memory_bank, dec_states, context_lengths=src_lengths)
        elif self.multimodal_model_type == 'src+img':
            dec_out, dec_out_imgs, dec_states, attn = self.model.decoder(
                    tgt_in, memory_bank, img_proj, dec_states,
                    context_lengths=src_lengths)
        else:
            raise Exception("Multi-modal odel type not implemented: %s"%(
                self.multimodal_model_type))

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores

    def _report_score(self, name, score_total, words_total):
        print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, score_total / words_total,
            name, math.exp(-score_total / words_total)))

    def _report_bleu(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        print()

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                      % (path, tgt_path, self.output),
                                      stdin=self.out_file,
                                      shell=True).decode("utf-8")

        print(">> " + res.strip())

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        print(res.strip())
