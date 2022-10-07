class EarlyStopping:
    def __init__(self, patience, ignore, method='consistency', delta=1e-4, fn='lt'):
        '''
        Args:
            patience : Integer, number of previous loss values stored at one time
            ignore : Integer, number of epochs to ignore
            method : String, 'consistency' breaks the training loop if the stored loss values are strictly increasing. 'slope' breaks the training loop if the regression line for the loss values equals or exceeds 0.
        '''
        self.patience = patience
        self.ignore = ignore
        self.method = method
        self.delta = delta
        self.fn = f'__{fn}__'
        self.history = []
        self.best_loss = 0
        self.methods = {
            'slope' : self.check_slope,
            'consistency' : self.check_consistency
        }
    
    def check(self, loss):
        '''
        Determines whether or not to break the training loop given a new loss value.

        Args:
            loss : Integer/Float, loss value to be considered
        
        Returns:
            Boolean : True means that the training loop should be broken
        '''
        self.history.append(loss)

        if self.is_stoppable():
            return True

        return False
    
    def is_stoppable(self):
        if len(self.history) > self.ignore:
            return self.methods[self.method]()

        self.best_loss = len(self.history) - 1
        return False

    def check_slope(self):
        raise NotImplementedError()
        x = [i for i in range(len(self.queue))]
        n = len(x)
        y = self.queue
        return (n*(sum([x[i]*y[i] for i in range(n)])) - sum(x)*sum(y))/(n*sum([v**2 for v in x]) - sum(x)**2) >= 0
    
    def check_consistency(self):
        last = self.history[-1]
        if getattr(last, self.fn)(self.history[self.best_loss]) and abs(last - self.history[self.best_loss]) > self.delta:
            self.best_loss = len(self.history) - 1
            return False
        return len(self.history) - 1 - self.best_loss >= self.patience
    
    def reset(self):
        '''
        Resets the EarlyStopping object
        '''
        self.__init__(self.patience, self.ignore, self.method, self.fn)
    
    def __str__(self):
        return f"History: {str(self.history)}; Best loss history index: {self.best_loss}"
    
    def __repr__(self):
        return self.__str__()