import random

def generate_data(num_samples):
  """
  Generates data for the file `data.csv`.

  Args:
    num_samples: The number of samples to generate.

  Returns:
    A list of lists, where each inner list contains a text and a label.
  """

  data = []
  for _ in range(num_samples):
    text = random.choice(["positive", "negative", "neutral"])
    label = "Positive" if text == "positive" else "Negative" if text == "negative" else "Neutral"
    data.append([text, label])

  return data

if __name__ == "__main__":
  data = generate_data(100)
  with open("data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)
