import json

"""
    json format
    {client_id : [eps1, eps2, ...], ...}
"""


# load file
def load_log_json(p):
    with open(p, 'r+') as f:
        data = json.load(f)
    f.close()
    return data


class DPLog:
    def __init__(self, idx, param_dict):
        self.IDX = idx
        self.CLIENT_EPS_DIR = {'params': param_dict}

    # log epsilon
    def save_epsilon(self, client_id, eps):
        if client_id not in self.CLIENT_EPS_DIR:
            self.CLIENT_EPS_DIR[client_id] = []
        self.CLIENT_EPS_DIR[client_id].append(eps)
        self.dump_json()

    def create_log_file(self):
        f = open('dp_eps_log/client_' + str(self.IDX) + '_dp_epsilon.json', 'w+')
        f.close()

    # save file
    def dump_json(self):
        with open('dp_eps_log/client_' + str(self.IDX) + '_dp_epsilon.json', 'w+') as f:
            json.dump(self.CLIENT_EPS_DIR, f)
        f.close()

    # load file
    def load_json(self):
        with open('dp_eps_log/client_' + str(self.IDX) + '_dp_epsilon.json', 'r+') as f:
            data = json.load(f)
        f.close()
        return data

    # debug
    def print_eps_list(self):
        print(self.CLIENT_EPS_DIR)
