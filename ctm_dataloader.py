import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import pre_processing
from typing import List


class CTMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)


def vocabulary_size_check(data: List[List[int]]) -> List[int]:
    """
    This function checks the vocabulary size for the dataset.
    """
    final_lst = []
    idx, step, vocab_size = 0, 2000, []
    for docs in data:
        idx += 1
        vocab_size.append(len(docs))
        if idx % step == 0:
            final_lst.append(max(vocab_size[idx - step: idx]))
    print(f"Vocabulary size for the dataset is {max(final_lst)}")
    return final_lst


def create_dataloader(documents: List[List[str]],
                      batch_size: int) -> DataLoader:
    """
    This function creates a dataloader for the dataset.
    """
    processed_data, vocab, words = pre_processing(documents, model_type = 'ctm')

    dataset = CTMDataset(processed_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, vocab, words


if __name__ == "__main__":
    news_data = pd.read_csv("newsgroups_data.csv")
    print("Shape of dataset:", news_data.shape)

    news_data = news_data.drop(columns=["Unnamed: 0"])

    documents = news_data.content
    target_labels = news_data.target
    target_names = news_data.target_names

    bag_of_words = pre_processing(documents)
    dataset = CTMDataset(bag_of_words)
    batch_size = 32

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for idx, data in enumerate(train_loader):
        print("Shape of data:", data.shape)
        print(data)
        break
