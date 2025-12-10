class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.randint(0, 2, ())) for _ in range(size)]
        # target now is scalar (0 or 1)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
