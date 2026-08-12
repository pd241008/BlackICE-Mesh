import urllib.request
import csv
import os
import numpy as np

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv"

CONTINUOUS_INDICES = [0, 4, 5, 7]
PROTOCOL_IDX = 1
SERVICE_IDX = 2
LABEL_IDX = 41

PROTOCOLS = ['tcp', 'udp', 'icmp']
SERVICES = ['http', 'private', 'domain_u', 'smtp', 'ftp_data', 'eco_i', 'ecr_i', 'other', 'telnet', 'finger', 'ftp']

def download_and_preprocess():
    os.makedirs("./data", exist_ok=True)
    train_path_raw = "./data/KDDTrain_raw.csv"
    test_path_raw = "./data/KDDTest_raw.csv"
    
    print("Downloading Train...")
    if not os.path.exists(train_path_raw):
        urllib.request.urlretrieve(TRAIN_URL, train_path_raw)
    
    print("Downloading Test...")
    if not os.path.exists(test_path_raw):
        urllib.request.urlretrieve(TEST_URL, test_path_raw)
        
    def parse_file(filepath):
        parsed = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 42:
                    continue
                # Continuous
                cont = [float(row[i]) for i in CONTINUOUS_INDICES]
                
                # Protocol One-hot
                prot = row[PROTOCOL_IDX]
                prot_vec = [1.0 if prot == p else 0.0 for p in PROTOCOLS]
                if sum(prot_vec) == 0:
                    prot_vec[0] = 1.0 # fallback
                    
                # Service One-hot
                svc = row[SERVICE_IDX]
                if svc not in SERVICES:
                    svc = 'other'
                svc_vec = [1.0 if svc == s else 0.0 for s in SERVICES]
                
                # Label
                label = 0.0 if row[LABEL_IDX] == 'normal' else 1.0
                
                row_vec = cont + prot_vec + svc_vec + [label]
                parsed.append(row_vec)
        return np.array(parsed)

    print("Parsing Train...")
    train_data = parse_file(train_path_raw)
    print("Parsing Test...")
    test_data = parse_file(test_path_raw)
    
    # Min-Max Scaling on continuous (first 4 columns)
    c_min = train_data[:, :4].min(axis=0)
    c_max = train_data[:, :4].max(axis=0)
    
    # Add epsilon
    denominator = (c_max - c_min) + 1e-8
    
    train_data[:, :4] = (train_data[:, :4] - c_min) / denominator
    test_data[:, :4] = (test_data[:, :4] - c_min) / denominator
    
    # Optional clamping for test set just in case
    test_data[:, :4] = np.clip(test_data[:, :4], 0.0, 1.0)
    
    print("Saving preprocessed files...")
    np.savetxt("./data/nsl-kdd-train.csv", train_data, delimiter=",", fmt="%.6f")
    np.savetxt("./data/nsl-kdd-test.csv", test_data, delimiter=",", fmt="%.6f")
    print(f"Done! Train shape: {train_data.shape}, Test shape: {test_data.shape}")

if __name__ == "__main__":
    download_and_preprocess()
