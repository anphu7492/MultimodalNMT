#!/usr/bin/env python

from __future__ import division, unicode_literals
import argparse
import onmt.opts
from onmt.translate.TranslatorMultimodal import make_translator


def main(opt):
    translator = make_translator(opt, report_score=True)
    translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate_mm.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    onmt.opts.translate_mm_opts(parser)

    opt = parser.parse_args()
    main(opt)
