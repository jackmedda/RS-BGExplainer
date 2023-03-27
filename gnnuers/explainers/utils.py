

class LRScaler(object):

    def __int__(self, optimizer, n_batches):
        self.lr_scaling = np.linspace(0.1, 1, n_batches)
        self.optimizer = optimizer

        self.orig_lr = [pg['lr'] for pg in self.optimizer.param_groups]

    def update(self, batch_idx):
        for pg_idx, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = orig_lr[pg_idx] * self.lr_scaling[batch_idx]

    def restore(self):
        for pg_idx, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.orig_lr[pg_idx]
