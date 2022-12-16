ATTEMPT_ID = '4'

TRAIN_MFCC_DIR = 'data/full/hw4p2/train-clean-100/mfcc/'
TRAIN_TRANSCRIPT_DIR = 'data/full/hw4p2/train-clean-100/transcript/raw/'

VAL_MFCC_DIR = 'data/full/hw4p2/dev-clean/mfcc/'
VAL_TRANSCRIPT_DIR = 'data/full/hw4p2/dev-clean/transcript/raw/'

TEST_MFCC_DIR = 'data/full/hw4p2/test-clean/mfcc/'
SUBMISSION_FILE = 'data/full/hw4p2/test-clean/transcript/random_submission.csv'


VOCAB = ['<sos>',
         'A',   'B',    'C',    'D',
         'E',   'F',    'G',    'H',
         'I',   'J',    'K',    'L',
         'M',   'N',    'O',    'P',
         'Q',   'R',    'S',    'T',
         'U',   'V',    'W',    'X',
         'Y',   'Z',    "'",    ' ',
         '<eos>']

VOCAB_MAP = {VOCAB[i]: i for i in range(0, len(VOCAB))}
SOS_TOKEN = VOCAB_MAP['<sos>']
EOS_TOKEN = VOCAB_MAP['<eos>']


CONFIG = {'batch_size': 128,
          'epochs': 10,
          'const_lr': 1e-3,

          'cosann_start_lr': 1e-4,
          'cosann_min_lr': 1e-6,

          'input_size': 15,
          'encoder_hidden_size': 512,
          'embed_size': 512,
          'decoder_hidden_size': 1024,
          'decoder_output_size': 256,
          'projection_size': 256}
