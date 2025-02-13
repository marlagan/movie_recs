import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import unique

import data_preprocessing as dp

class MovieRecsModel(nn.Module):

    def __init__(self, dataset):
        super(MovieRecsModel, self).__init__()

        num_desc = unique(dataset['description'])
        num_languages = unique(dataset['org_language'])
        num_release_date = unique(dataset['release_date'])
        num_average = unique(dataset['vote_average'])
        num_runtime = unique(dataset['runtime'])

        desc_embedding = nn.Embedding(num_desc, 128)
        lang_embedding = nn.Embedding(num_languages, 128)
        release_embedding = nn.Embedding(num_release_date, 128)
        average_embedding = nn.Embedding(num_average, 128)
        num_runtime = nn.Embedding(num_runtime, 128)

        self.out = nn.Linear(64, 1)





