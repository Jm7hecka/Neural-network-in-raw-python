from tqdm import tqdm
import time

class ProgressBar:
    def __init__(self, epoch, total_data):
        self.loss = 0
        self.accuracy = 0
        self.epochs = epoch
        self.total_data = total_data

    def run(self):
        
        self.pbar = tqdm(total=self.total_data, desc=' ', unit='data', dynamic_ncols=True, bar_format="[{n_fmt}/{total_fmt}] |{bar}| {elapsed} | {unit} ")

        def update(loss, accuracy):
            self.pbar.unit = f'Loss: {loss:.4f} Accuracy: {accuracy:.4f}'
            self.pbar.update(1)
            if self.pbar.n == self.total_data:
                tqdm.close(self.pbar)
        return update

