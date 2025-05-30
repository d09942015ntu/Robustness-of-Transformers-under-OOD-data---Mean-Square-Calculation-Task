import json
import random

random.seed(42)  # Fixed seed for reproducibility

# Generate float with decimal part >= 0.5 and range [offset, offset+10]
def generate_shifted_float(offset):
    int_part = random.randint(offset, offset + 9)
    dec_part = random.uniform(0.5, 0.999)
    return round(int_part + dec_part, 3)

# Generate one sample
def generate_sample(offset):
    x = [generate_shifted_float(offset) for _ in range(4)]
    input_str = " ".join(f"{v:.3f}" for v in x)
    mean = round(sum(x) / 4, 3)
    sq = round(mean ** 2, 3)
    output_str = f"{mean:.3f} {sq:.3f}"
    return {"input": input_str, "output": output_str}

# Generate datasets for i in 0~5
def generate_datasets():
    for i in range(6):
        path = f"ood_testing_data_shift{i}.jsonl"
        with open(path, 'w') as f:
            for _ in range(1000):
                sample = generate_sample(i)
                f.write(json.dumps(sample) + "\n")
        print(f"✅ Generated {path}")

if __name__ == '__main__':
    generate_datasets()