import json
import random

random.seed(42)  # Fixed seed for reproducibility

# Generate float with decimal part < 0.5
def generate_limited_float():
    int_part = random.randint(0, 9)
    dec_part = random.uniform(0.0, 0.499)
    return round(int_part + dec_part, 3)

# Generate one sample
def generate_sample():
    x = [generate_limited_float() for _ in range(4)]
    input_str = " ".join(f"{v:.3f}" for v in x)
    mean = round(sum(x) / 4, 3)
    sq = round(mean ** 2, 3)
    output_str = f"{mean:.3f} {sq:.3f}"
    return {"input": input_str, "output": output_str}

# Generate dataset
def generate_dataset(path, n):
    with open(path, 'w') as f:
        for _ in range(n):
            sample = generate_sample()
            f.write(json.dumps(sample) + "\n")
    print(f"✅ Generated {path} with {n} samples")

if __name__ == '__main__':
    generate_dataset("id_training_data.jsonl", 1_000_000)