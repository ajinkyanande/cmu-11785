import os
from tqdm import tqdm
import numpy as np
import torch
import torchaudio

from params import *


class AudioDatasetTrain(torch.utils.data.Dataset):
    '''
    DOCSTRING
    '''

    def __init__(self, mfcc_dir, transcript_dir):

        # mfcc and transcript file names
        mfcc_files = sorted(os.listdir(mfcc_dir))
        transcript_files = sorted(os.listdir(transcript_dir))

        # dataset length
        assert len(mfcc_files) == len(transcript_files)
        self.length = len(mfcc_files)

        self.mfccs = []
        self.transcripts = []

        for i in tqdm(range(self.length), total=self.length):

            mfcc = np.load(mfcc_dir+mfcc_files[i])
            transcript_str = np.load(transcript_dir+transcript_files[i])

            # cepstral normalization of mfcc
            mfcc = mfcc - np.mean(mfcc, axis=0, keepdims=True)
            mfcc = mfcc / np.std(mfcc, axis=0, keepdims=True)

            # letters to dictionary indexes
            transcript_int = np.array([VOCAB_MAP[p] for p in transcript_str])

            self.mfccs.append(mfcc)
            self.transcripts.append(transcript_int)

    def __len__(self):

        return self.length

    def __getitem__(self, ind):

        mfcc = torch.FloatTensor(self.mfccs[ind])
        transcript = torch.LongTensor(self.transcripts[ind])

        masking_time = torchaudio.transforms.TimeMasking(time_mask_param=5)
        mfcc_time_masked = masking_time(mfcc.unsqueeze(0)).squeeze(0)

        masking_freq = torchaudio.transforms.FrequencyMasking(freq_mask_param=50)
        mfcc_time_freq_masked = masking_freq(mfcc_time_masked.unsqueeze(0)).squeeze(0)

        return mfcc_time_freq_masked, transcript

    def collate_fn(batch):

        # batch of input mfcc coefficients and transcripts
        batch_mfcc = [b[0] for b in batch]
        batch_transcript = [b[1] for b in batch]

        # pad mfccs of batch to make of same length
        lengths_mfcc = [len(b) for b in batch_mfcc]
        batch_mfcc_pad = torch.nn.utils.rnn.pad_sequence(batch_mfcc, batch_first=True)

        # pad mfccs of batch to make of same length
        lengths_transcript = [len(b) for b in batch_transcript]
        batch_transcript_pad = torch.nn.utils.rnn.pad_sequence(batch_transcript, batch_first=True)

        return batch_mfcc_pad, torch.tensor(lengths_mfcc), batch_transcript_pad, torch.tensor(lengths_transcript)


class AudioDatasetVal(torch.utils.data.Dataset):
    '''
    DOCSTRING
    '''

    def __init__(self, mfcc_dir, transcript_dir):

        # mfcc and transcript file names
        mfcc_files = sorted(os.listdir(mfcc_dir))
        transcript_files = sorted(os.listdir(transcript_dir))

        # dataset length
        assert len(mfcc_files) == len(transcript_files)
        self.length = len(mfcc_files)

        self.mfccs = []
        self.transcripts = []

        for i in tqdm(range(self.length), total=self.length):

            mfcc = np.load(mfcc_dir+mfcc_files[i])
            transcript_str = np.load(transcript_dir+transcript_files[i])

            # cepstral normalization of mfcc
            mfcc = mfcc - np.mean(mfcc, axis=0, keepdims=True)
            mfcc = mfcc / np.std(mfcc, axis=0, keepdims=True)

            # letters to dictionary indexes
            transcript_int = np.array([VOCAB_MAP[p] for p in transcript_str])

            self.mfccs.append(mfcc)
            self.transcripts.append(transcript_int)

    def __len__(self):

        return self.length

    def __getitem__(self, ind):

        return torch.FloatTensor(self.mfccs[ind]), torch.LongTensor(self.transcripts[ind])

    def collate_fn(batch):

        # batch of input mfcc coefficients and transcripts
        batch_mfcc = [b[0] for b in batch]
        batch_transcript = [b[1] for b in batch]

        # pad mfccs of batch to make of same length
        lengths_mfcc = [len(b) for b in batch_mfcc]
        batch_mfcc_pad = torch.nn.utils.rnn.pad_sequence(batch_mfcc, batch_first=True)

        # pad mfccs of batch to make of same length
        lengths_transcript = [len(b) for b in batch_transcript]
        batch_transcript_pad = torch.nn.utils.rnn.pad_sequence(batch_transcript, batch_first=True)

        return batch_mfcc_pad, torch.tensor(lengths_mfcc), batch_transcript_pad, torch.tensor(lengths_transcript)


class AudioDatasetTest(torch.utils.data.Dataset):
    '''
    DOCSTRING
    '''

    def __init__(self, mfcc_dir):

        # mfcc file names
        mfcc_files = sorted(os.listdir(mfcc_dir))

        # dataset length
        self.length = len(mfcc_files)

        self.mfccs = []

        for i in tqdm(range(self.length), total=self.length):

            mfcc = np.load(mfcc_dir+mfcc_files[i])

            # cepstral normalization of mfcc
            mfcc = mfcc - np.mean(mfcc, axis=0, keepdims=True)
            mfcc = mfcc / np.std(mfcc, axis=0, keepdims=True)

            self.mfccs.append(mfcc)

    def __len__(self):

        return self.length

    def __getitem__(self, ind):

        return torch.FloatTensor(self.mfccs[ind])

    def collate_fn(batch):

        # batch of input mfcc coefficients and transcripts
        batch_mfcc = [b for b in batch]

        # pad mfccs of batch to make of same length
        lengths_mfcc = [len(b) for b in batch_mfcc]
        batch_mfcc_pad = torch.nn.utils.rnn.pad_sequence(batch_mfcc, batch_first=True)

        return batch_mfcc_pad, torch.tensor(lengths_mfcc)


def get_loaders():

    train_data = AudioDatasetTrain(TRAIN_MFCC_DIR, TRAIN_TRANSCRIPT_DIR)
    val_data = AudioDatasetVal(VAL_MFCC_DIR, VAL_TRANSCRIPT_DIR)
    test_data = AudioDatasetTest(TEST_MFCC_DIR)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=CONFIG['batch_size'],
                                               collate_fn=AudioDatasetTrain.collate_fn,
                                               pin_memory=True,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=CONFIG['batch_size'],
                                             collate_fn=AudioDatasetVal.collate_fn,
                                             pin_memory=True,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=CONFIG['batch_size'],
                                              collate_fn=AudioDatasetTest.collate_fn,
                                              pin_memory=True,
                                              shuffle=False)

    print('\nbatch size: ', CONFIG['batch_size'])
    print('train dataset samples:', len(train_data), 'batches:', len(train_loader))
    print('val dataset samples:', len(val_data), 'batches:', len(val_loader))
    print('test dataset samples:', len(test_data), 'batches:', len(test_loader))

    return train_loader, val_loader, test_loader
