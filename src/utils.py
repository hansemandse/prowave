import numpy as np
import pandas as pd

def setify(data):
    """
    Produces summary statistics for the provided datasets.

    Args:
     `data`: a list of datasets

    Returns a tuple of three lists;
     `uniques` representing sets of clans and families in each of the datasets
     `perclan` representing tuples of clan ID and protein count
     `perfam`  representing tuples of family ID and protein count
    """
    uniques = [[set() for _ in range(len(data))] for _ in range(2)]
    perclan = [{} for _ in range(len(data))]
    perfam  = [{} for _ in range(len(data))]
    for index, d in enumerate(data):
        for i in range(len(d)):
            # Fetch this entry
            row  = d[i]
            clan = row[2]
            fam  = row[3]

            # Add clan and family IDs to sets
            uniques[0][index].add(clan) # add clan
            uniques[1][index].add(fam) # add family

            # Count proteins in this clan
            if clan not in perclan[index]:
                perclan[index][clan] = 1
            else:
                perclan[index][clan] += 1

            # Count proteins in this family
            if fam not in perfam[index]:
                perfam[index][fam] = 1
            else:
                perfam[index][fam] += 1

    return uniques, [x.items() for x in perclan], [x.items() for x in perfam]

# The following is taken from EXE 5.1 RNN
def one_hot_encode(idx, vocab_size):
    """
    One-hot encodes a single word given its index and the size of the vocabulary.
    
    Args:
     `idx`: the index of the given word
     `vocab_size`: the size of the vocabulary
    
    Returns a 1-D numpy array of length `vocab_size`.
    """
    # Initialize the encoded array
    one_hot = np.zeros(vocab_size)
    
    # Set the appropriate element to one
    one_hot[idx] = 1.0

    return one_hot

def one_hot_encode_sequence(sequence, vocab_size):
    """
    One-hot encodes a sequence of words given a fixed vocabulary size.
    
    Args:
     `sentence`: a list of words to encode
     `vocab_size`: the size of the vocabulary
     
    Returns a 3-D numpy array of shape (num words, vocab size, 1).
    """
    # Encode each word in the sentence
    encoding = np.array([one_hot_encode(word, vocab_size) for word in sequence])

    # Reshape encoding s.t. it has shape (num words, vocab size, 1)
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1)
    
    return encoding
# End of EXE 5.1 RNN

def get_data(dataset, num):
    """
    Fetches data for `num` clans in the given dataset

    Args:
     `dataset`: an LMDB dataset
     `num`: a number of clans to collect data for

    Returns a DataFrame object with the fetched data
    """
    res = {'Sequence' : [], 'Clan ID' : [], 'Family ID' : []}
    clans = set(range(num))
    bound = len(dataset)//100
    print(f'Bound = {bound}')
    print('Process ['+' '*100+']')
    for i in range(len(dataset)):
        # Fetch this entry
        row = dataset[i]
        clan = row[2]

        # If the clan ID is not in the set of wanted IDs, skip this entry
        if clan not in clans:
            continue

        # Otherwise, store the data
        res['Sequence'].append(row[0].astype('int16'))
        res['Clan ID'].append(clan.astype('int16'))
        res['Family ID'].append(row[3].astype('int16'))

        # Update process
        if (i % bound == 0):
            print(f"\033[FProcess [{'='*(i//bound)}{' '*(100-i//bound)}]")

    return pd.DataFrame.from_dict(res)

def get_data_input(dataset, num):
    """
    Fetches data for input

    Args:
     `dataset`: a Panda Dataset
     `num`: a number for how many amiooacids should go into the input

    Returns a DataFrame object with the fetched data
    """
    res = {'Sequence' : [], 'Clan ID' : [], 'Family ID' : []}
    
    for i in range(len(dataset)):
        # Fetch this entry
        Clan = dataset['Clan ID'].iloc[i]
        Family = dataset['Family ID'].iloc[i]
        sequence = dataset['Sequence'].iloc[i]
        # Otherwise, store the data
        res['Sequence'].append(sequence.astype('int16'))
        res['Clan ID'].append(Clan.astype('int16'))
        res['Family ID'].append(Family.astype('int16'))

    return pd.DataFrame.from_dict(res)


def get_data_output(dataset):
    """
    Fetches data for output

    Args:
     `dataset`: a Panda Dataset

    Returns a DataFrame object with the fetched data
    """
    res = {'Sequence' : []}
    
    for i in range(len(dataset)):
        # Fetch this entry
        sequence = ((dataset['Sequence'].iloc[i]))
        # Otherwise, store the data
        res['Sequence'].append(sequence.astype('int16'))

    return pd.DataFrame.from_dict(res)
