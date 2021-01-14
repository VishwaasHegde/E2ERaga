# from crepe.cli import run
import core
import argparse
import sys

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--runtime', default=False,
                            help='records audio for the given amount of duration')
    arg_parser.add_argument('--duration', default=120,
                            help='sets the duration for recording in seconds')
    arg_parser.add_argument('--tradition', default=None,
                            help='sets the tradition - hindustani/carnatic')
    # #
    p_args = arg_parser.parse_args()
    #
    if p_args.tradition is None:
        sys.exit("Please set the tradition - hindustani/carnatic")
    #
    if p_args.runtime:
        core.predict_run_time(tradition=p_args.tradition)
    # if
    # import os
    # filename = [os.path.join(os.path.dirname(__file__), 'tests/sample_data/gap_2.wav')]
    # run(filename, viterbi=False)
    # core.train(task='raga', tradition='hindustani')
    # core.train(task='tonic', tradition='hindustani')
    # core.test(task='raga', tradition='hindustani')
    # core.test_pitch(task='pitch', tradition='carnatic')
    # core.train_pitch(tradition='hindustani')
    # core.cache_cqt(task='raga', tradition='hindustani')
    # core.predict_run_time(tradition='hindustani')
