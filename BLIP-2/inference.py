import matplotlib.pyplot as plt
from torchvision import transforms



def plot_image_with_prediction(image_tensor, question, answer, prediction):
  """Plots an image along with the question, correct answer, and model prediction.

  Args:
      image_tensor: The image tensor.
      question: The question string.
      answer: The correct answer string.
      prediction: The model's prediction string.
  """
  # Convert the image tensor to a PIL Image
  t = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                             std=[1/0.229, 1/0.224, 1/0.225])

  image = transforms.ToPILImage()(t(image_tensor))

  # Create the plot
  plt.figure(figsize=(8, 6))
  plt.imshow(image)
  plt.axis('off')  # Hide axis ticks and labels

  # Add text annotations
  plt.text(0, -10, f"Question: {question}", fontsize=10, color='black', ha='left', va='top')
  plt.text(0, 5, f"Correct Answer: {answer}", fontsize=10, color='red', ha='left', va='top')
  plt.text(0, 10, f"Prediction: {prediction}", fontsize=10, color='red', ha='left', va='top')

  # Display the plot
  plt.show()

# Example usage (within your existing code, after the loops):

for i in val_loader:
    preds = model.generate(i[0], i[1])
    questions = i[1]
    answers = i[2]
    images = i[0]

    for j in range(min(len(preds), len(questions), len(answers), len(images))): # Ensure indices are valid
      question_tokens = [data.rev_vocab[tok.item()] for tok in questions[j] if tok not in [0, 1, 2]]
      question = " ".join(question_tokens)
      answer_tokens = [data.rev_vocab[tok.item()] for tok in answers[j] if tok not in [0, 1, 2]]
      answer = " ".join(answer_tokens)
      
      plot_image_with_prediction(images[j], question, answer, preds[j][5:])
      if j==32:
        break

    break
