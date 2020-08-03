# Customize adding backdoor to transforms that speeds up preprocess.
class AddTrigger(object):
    def __init__(self, trigger_loc, trigger_ptn):
        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        for i, (m, n) in enumerate(self.trigger_loc):
            img[m, n, :] = self.trigger_ptn[i]  # add trigger
        return img

